"""
Athlete Injury Risk Predictor — Feature Engineering
Builds a leakage-safe, athlete-level feature table from raw wearable + training data.

Observation window: days 1-30 (features only)
Risk window:        days 31-60 (label window, never touched here)
"""
import pandas as pd
import numpy as np

DATA_DIR = "/mnt/user-data/uploads/"
START = pd.Timestamp("2026-01-05")   # day 1
CUTOFF = pd.Timestamp("2026-02-03")  # day 30 (last day of observation window)


def load_raw():
    meta = pd.read_csv(DATA_DIR + "athlete_metadata.csv")
    labels = pd.read_csv(DATA_DIR + "train_labels.csv")
    ts = pd.read_csv(DATA_DIR + "training_sessions.csv", parse_dates=["date"])
    da = pd.read_csv(DATA_DIR + "dailyActivity_merged.csv", parse_dates=["ActivityDate"])
    sl = pd.read_csv(DATA_DIR + "sleepDay_merged.csv", parse_dates=["SleepDay"])
    wl = pd.read_csv(DATA_DIR + "weightLogInfo_merged.csv", parse_dates=["Date"])
    hr = pd.read_csv(DATA_DIR + "hourlyHeartrate_merged.csv", parse_dates=["ActivityHour"])
    return meta, labels, ts, da, sl, wl, hr


def restrict_to_observation_window(ts, da, sl, wl, hr):
    """Critical leakage guard: drop any row outside days 1-30 before aggregating anything."""
    ts = ts[(ts.date >= START) & (ts.date <= CUTOFF)]
    da = da[(da.ActivityDate >= START) & (da.ActivityDate <= CUTOFF)]
    sl = sl[(sl.SleepDay >= START) & (sl.SleepDay <= CUTOFF)]
    wl = wl[(wl.Date >= START) & (wl.Date <= CUTOFF)]
    hr = hr[(hr.ActivityHour >= START) & (hr.ActivityHour <= CUTOFF)]
    return ts, da, sl, wl, hr


def aggregate_hourly_hr_to_daily(hr):
    hr = hr.copy()
    hr["day"] = hr.ActivityHour.dt.floor("D")
    return hr.groupby(["Id", "day"]).agg(
        avg_hr=("AvgHeartRate", "mean"),
        min_hr=("MinHeartRate", "min"),   # resting-HR proxy
        max_hr=("MaxHeartRate", "max"),
    ).reset_index()


def build_daily_unified(meta, ts, da, sl, hr_daily):
    ts = ts.copy()
    ts["duration_hr"] = (ts.end_hour - ts.start_hour).clip(lower=0)
    ts_daily = ts.groupby(["athlete_id", "date"]).agg(
        n_sessions=("session_id", "count"),
        session_duration_hr=("duration_hr", "sum"),
    ).reset_index().rename(columns={"athlete_id": "Id", "date": "day"})

    da_r = da.rename(columns={"ActivityDate": "day"})
    sl_r = sl.rename(columns={"SleepDay": "day"})[["Id", "day", "TotalMinutesAsleep", "TotalTimeInBed"]]

    all_ids = meta.athlete_id.unique()
    all_days = pd.date_range(START, CUTOFF, freq="D")
    grid = pd.MultiIndex.from_product([all_ids, all_days], names=["Id", "day"]).to_frame(index=False)

    daily = (
        grid.merge(da_r, on=["Id", "day"], how="left")
        .merge(sl_r, on=["Id", "day"], how="left")
        .merge(hr_daily, on=["Id", "day"], how="left")
        .merge(ts_daily, on=["Id", "day"], how="left")
    )
    daily["n_sessions"] = daily["n_sessions"].fillna(0)
    daily["session_duration_hr"] = daily["session_duration_hr"].fillna(0)
    daily["load"] = daily["session_duration_hr"]  # simple training-load proxy
    return daily


def window_feats(df, n_days, suffix):
    w = df[df.days_before_cutoff < n_days]
    return w.groupby("Id").agg(**{
        f"steps_mean_{suffix}": ("TotalSteps", "mean"),
        f"steps_std_{suffix}": ("TotalSteps", "std"),
        f"sedentary_mean_{suffix}": ("SedentaryMinutes", "mean"),
        f"very_active_mean_{suffix}": ("VeryActiveMinutes", "mean"),
        f"calories_mean_{suffix}": ("Calories", "mean"),
        f"sleep_mean_{suffix}": ("TotalMinutesAsleep", "mean"),
        f"sleep_std_{suffix}": ("TotalMinutesAsleep", "std"),
        f"time_in_bed_mean_{suffix}": ("TotalTimeInBed", "mean"),
        f"avg_hr_mean_{suffix}": ("avg_hr", "mean"),
        f"min_hr_mean_{suffix}": ("min_hr", "mean"),
        f"max_hr_mean_{suffix}": ("max_hr", "mean"),
        f"n_sessions_sum_{suffix}": ("n_sessions", "sum"),
        f"load_sum_{suffix}": ("load", "sum"),
        f"load_mean_{suffix}": ("load", "mean"),
    }).reset_index()


def build_feature_table():
    meta, labels, ts, da, sl, wl, hr = load_raw()
    ts, da, sl, wl, hr = restrict_to_observation_window(ts, da, sl, wl, hr)
    hr_daily = aggregate_hourly_hr_to_daily(hr)
    daily = build_daily_unified(meta, ts, da, sl, hr_daily)
    daily = daily.sort_values(["Id", "day"])
    daily["days_before_cutoff"] = (CUTOFF - daily["day"]).dt.days

    f7 = window_feats(daily, 7, "7d")
    f14 = window_feats(daily, 14, "14d")
    f30 = window_feats(daily, 30, "30d")

    feat = meta[["athlete_id"]].rename(columns={"athlete_id": "Id"})
    feat = feat.merge(f7, on="Id", how="left").merge(f14, on="Id", how="left").merge(f30, on="Id", how="left")

    feat["acwr"] = (feat["load_mean_7d"] / feat["load_mean_30d"].replace(0, np.nan)).fillna(0)
    feat["sleep_deficit_7d"] = 480 - feat["sleep_mean_7d"]
    feat["sleep_change_7v30"] = feat["sleep_mean_7d"] - feat["sleep_mean_30d"]
    feat["resting_hr_change_7v30"] = feat["min_hr_mean_7d"] - feat["min_hr_mean_30d"]
    feat["load_change_7v30"] = feat["load_mean_7d"] - feat["load_mean_30d"]
    feat["steps_change_7v30"] = feat["steps_mean_7d"] - feat["steps_mean_30d"]

    feat = feat.merge(meta.rename(columns={"athlete_id": "Id"}), on="Id", how="left")
    feat = feat.merge(
        labels.rename(columns={"athlete_id": "Id"})[["Id", "injured_in_risk_window"]],
        on="Id", how="left",
    )

    # Also export daily trend data for dashboard charts
    daily[["Id", "day", "TotalSteps", "TotalMinutesAsleep", "avg_hr", "min_hr", "load"]].to_csv(
        "daily_trend.csv", index=False
    )
    return feat


if __name__ == "__main__":
    feat = build_feature_table()
    feat.to_csv("feature_table.csv", index=False)
    print(f"Feature table saved: {feat.shape}")
    print(feat["injured_in_risk_window"].value_counts(normalize=True))
