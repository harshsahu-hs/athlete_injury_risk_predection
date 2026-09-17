# Athlete Injury Risk Predictor

AI-powered early-warning system that analyzes 30 days of training, activity, sleep,
and heart-rate data to estimate an athlete's injury risk over the following 30 days,
explains the main contributing factors, and gives coaches a team-level view.

## Problem
Predict `injured_in_risk_window` (binary) using only observation-window data
(days 1–30) to avoid leakage from the risk window (days 31–60).

## Data
- athlete_metadata.csv — demographics, sport, position, prior injuries
- training_sessions.csv — session-level training log
- dailyActivity_merged.csv, sleepDay_merged.csv, weightLogInfo_merged.csv — daily wearable data
- hourlyHeartrate_merged.csv — aggregated to daily resting/avg/max HR
- train_labels.csv — ground truth: injured_in_risk_window, onset_day_offset, recovery_duration

## Pipeline
1. `feature_engineering.py` — builds 7d/14d/30d rolling features, ACWR (acute:chronic
   workload ratio), sleep/HR/load week-over-month deltas → `feature_table.csv`
2. EDA — top signals are *changes* in the final week vs the monthly baseline
   (activity spike, sleep drop, resting HR rise), not steady-state load levels.
3. `train_model.py` — Logistic Regression, Random Forest, HistGradientBoosting
   compared on ROC-AUC / PR-AUC (imbalance-aware); HistGradientBoosting selected.
4. Explainability — permutation importance + per-athlete z-score deviation
   from population norms on the top model features (SHAP-equivalent, no extra deps).
5. `app.py` — Streamlit dashboard: athlete view (risk score, top factors, 30-day
   trend charts) and team view (roster sorted by risk tier).

## Model performance (held-out test set, n=600)
- ROC-AUC: 0.78, PR-AUC: 0.77
- At the 0.35 review threshold: 61% recall / 69% precision on the injury class
  (tuned for recall since missed injuries are the costly error for a screening tool)

## Risk tiers
HIGH ≥35%, MODERATE 20–35%, LOW <20% — product thresholds tuned against validation
data, not medical cutoffs.

## Run it
```
pip install -r requirements.txt
streamlit run app.py
```

## Framing
This is a screening / decision-support tool for coaching and medical staff —
not a diagnostic system.
