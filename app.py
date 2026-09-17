import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Athlete Injury Risk Predictor", layout="wide")

# ---------------- Load artifacts ----------------
@st.cache_resource
def load_artifacts():
    pipe = joblib.load("model.pkl")
    fmeta = joblib.load("feature_meta.pkl")
    return pipe, fmeta

@st.cache_data
def load_data():
    preds = pd.read_csv("full_predictions.csv")
    daily = pd.read_csv("daily_trend.csv", parse_dates=["day"])
    importance = pd.read_csv("feature_importance.csv")
    return preds, daily, importance

pipe, fmeta = load_artifacts()
preds, daily, importance = load_data()

TOP_FEATURES = importance.sort_values("importance", ascending=False).head(10)["feature"].tolist()
NUM_TOP = [f for f in TOP_FEATURES if f in preds.select_dtypes(include=np.number).columns]
POP_MEAN = preds[NUM_TOP].mean()
POP_STD = preds[NUM_TOP].std()

def risk_color(tier):
    return {"HIGH": "🔴", "MODERATE": "🟠", "LOW": "🟢"}.get(tier, "⚪")

def explain_athlete(row, n=4):
    z = ((row[NUM_TOP] - POP_MEAN) / POP_STD).abs().sort_values(ascending=False)
    out = []
    for f in z.index[:n]:
        direction = "above" if row[f] > POP_MEAN[f] else "below"
        out.append((f, row[f], direction, z[f]))
    return out

FEATURE_LABELS = {
    "steps_std_7d": "Step count variability (7d)",
    "steps_change_7v30": "Step count change (week vs month)",
    "resting_hr_change_7v30": "Resting HR change (week vs month)",
    "sleep_change_7v30": "Sleep change (week vs month)",
    "very_active_mean_7d": "Very active minutes/day (7d avg)",
    "steps_std_30d": "Step count variability (30d)",
    "calories_mean_7d": "Calories burned/day (7d avg)",
    "very_active_mean_30d": "Very active minutes/day (30d avg)",
    "sleep_std_7d": "Sleep variability (7d)",
}

st.sidebar.title("🏃 Athlete Injury AI")
page = st.sidebar.radio("View", ["Athlete Dashboard", "Team Dashboard"])

# ================= ATHLETE DASHBOARD =================
if page == "Athlete Dashboard":
    athlete_id = st.sidebar.selectbox("Select athlete", preds["Id"].sort_values())
    row = preds[preds["Id"] == athlete_id].iloc[0]

    st.title(f"Athlete {athlete_id} — {row['sport']}")
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.metric("Injury Risk Score", f"{row['risk_score']*100:.0f}%")
        st.markdown(f"### {risk_color(row['risk_tier'])} {row['risk_tier']} RISK")

    with col2:
        st.metric("Age", int(row["age"]))
        st.metric("Position", row["position"])
        st.metric("Prior season injuries", int(row["prior_season_injury_count"]))

    with col3:
        st.subheader("Main contributing factors")
        for feat, val, direction, z in explain_athlete(row):
            label = FEATURE_LABELS.get(feat, feat)
            arrow = "⬆️" if direction == "above" else "⬇️"
            st.write(f"{arrow} **{label}** — {val:.1f} ({direction} typical, z={z:.2f})")

    st.divider()
    st.subheader("30-day trend (observation window)")
    ath_daily = daily[daily["Id"] == athlete_id].sort_values("day")

    t1, t2 = st.columns(2)
    with t1:
        st.caption("Steps / day")
        st.line_chart(ath_daily.set_index("day")["TotalSteps"])
        st.caption("Sleep (minutes) / night")
        st.line_chart(ath_daily.set_index("day")["TotalMinutesAsleep"])
    with t2:
        st.caption("Avg heart rate / day")
        st.line_chart(ath_daily.set_index("day")["avg_hr"])
        st.caption("Training load (session-hours) / day")
        st.line_chart(ath_daily.set_index("day")["load"])

    st.caption(
        "This score is a decision-support estimate based on recent training and recovery "
        "patterns — not a medical diagnosis. Review with medical/coaching staff before acting."
    )

# ================= TEAM DASHBOARD =================
else:
    st.title("👥 Team Health Overview")

    teams = sorted(preds["team_id"].dropna().unique())
    team_choice = st.sidebar.selectbox("Team", ["All"] + teams)
    view = preds if team_choice == "All" else preds[preds["team_id"] == team_choice]

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 High risk", int((view["risk_tier"] == "HIGH").sum()))
    c2.metric("🟠 Moderate risk", int((view["risk_tier"] == "MODERATE").sum()))
    c3.metric("🟢 Low risk", int((view["risk_tier"] == "LOW").sum()))

    st.divider()
    display_cols = ["Id", "sport", "position", "prior_season_injury_count", "risk_score", "risk_tier"]
    table = view[display_cols].sort_values("risk_score", ascending=False).copy()
    table["risk_score"] = (table["risk_score"] * 100).round(0).astype(int).astype(str) + "%"
    table["status"] = table["risk_tier"].apply(risk_color)
    table = table[["Id", "sport", "position", "prior_season_injury_count", "risk_score", "status", "risk_tier"]]
    table.columns = ["Athlete", "Sport", "Position", "Prior Injuries", "Risk", "", "Tier"]
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.caption(
        "Risk tiers — HIGH ≥35%, MODERATE 20–35%, LOW <20% — tuned against validation data "
        "for recall on the injury class. These are review thresholds, not clinical cutoffs."
    )
