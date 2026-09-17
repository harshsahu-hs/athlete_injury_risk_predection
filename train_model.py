"""
Athlete Injury Risk Predictor — Model training, evaluation, explainability.
Run after feature_engineering.py has produced feature_table.csv.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

REVIEW_THRESHOLD = 0.35   # tuned for recall on the injury class; see README


def train_and_evaluate():
    feat = pd.read_csv("feature_table.csv")
    y = feat["injured_in_risk_window"]
    X = feat.drop(columns=["injured_in_risk_window", "Id"])

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    candidates = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=8, class_weight="balanced_subsample", random_state=42
        ),
        "hist_gb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, random_state=42),
    }

    best_name, best_pipe, best_pr_auc = None, None, -1
    for name, clf in candidates.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)
        val_proba = pipe.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_proba)
        pr_auc = average_precision_score(y_val, val_proba)
        print(f"{name}: val ROC-AUC={auc:.3f}  PR-AUC={pr_auc:.3f}")
        if pr_auc > best_pr_auc:
            best_name, best_pipe, best_pr_auc = name, pipe, pr_auc

    print(f"\nBest model: {best_name}")
    test_proba = best_pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= REVIEW_THRESHOLD).astype(int)
    print(f"\n--- TEST performance @ threshold={REVIEW_THRESHOLD} ---")
    print("ROC-AUC:", roc_auc_score(y_test, test_proba))
    print("PR-AUC:", average_precision_score(y_test, test_proba))
    print(classification_report(y_test, test_pred, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_test, test_pred))

    # Explainability: permutation importance (SHAP-free, model-agnostic)
    result = permutation_importance(
        best_pipe, X_test, y_test, n_repeats=10, random_state=42, scoring="average_precision", n_jobs=-1
    )
    importance = pd.DataFrame(
        {"feature": X_test.columns, "importance": result.importances_mean}
    ).sort_values("importance", ascending=False)
    importance.to_csv("feature_importance.csv", index=False)

    joblib.dump(best_pipe, "model.pkl")
    joblib.dump({"num_cols": num_cols, "cat_cols": cat_cols}, "feature_meta.pkl")
    X_test.assign(y_true=y_test.values, risk_score=test_proba).to_csv("test_predictions.csv", index=False)

    # Score the full roster for the dashboard
    X_all = feat.drop(columns=["injured_in_risk_window", "Id"])
    feat["risk_score"] = best_pipe.predict_proba(X_all)[:, 1]
    feat["risk_tier"] = pd.cut(
        feat["risk_score"], bins=[-0.01, 0.20, 0.35, 1.01], labels=["LOW", "MODERATE", "HIGH"]
    )
    feat.to_csv("full_predictions.csv", index=False)

    print("\nSaved model.pkl, feature_meta.pkl, feature_importance.csv, test_predictions.csv, full_predictions.csv")
    return best_pipe, importance


if __name__ == "__main__":
    train_and_evaluate()
