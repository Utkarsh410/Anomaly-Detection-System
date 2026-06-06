import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

df = pd.read_csv("creditcard.csv")
print(f"Dataset loaded: {len(df):,} rows  |  fraud rate: {df['Class'].mean():.4%}")

# Scale Amount and persist scaler so inference applies identical transformation
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ── XGBoost (supervised) ────────────────────────────────────────────────────
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
)
xgb_model.fit(X_train, y_train)

# ── Isolation Forest (unsupervised) ────────────────────────────────────────
iso_model = IsolationForest(contamination=0.002, random_state=RANDOM_STATE)
iso_model.fit(X_train)

# ── Evaluation ─────────────────────────────────────────────────────────────
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
iso_preds = iso_model.predict(X_test)
iso_scores = np.where(iso_preds == -1, 1.0, 0.0)

ensemble_scores = 0.7 * xgb_probs + 0.3 * iso_scores
ensemble_preds = (ensemble_scores > 0.5).astype(int)

print("\n── Ensemble Classification Report ──")
print(classification_report(y_test, ensemble_preds, target_names=["Normal", "Fraud"]))
print(f"XGBoost ROC-AUC : {roc_auc_score(y_test, xgb_probs):.4f}")
print(f"Ensemble ROC-AUC: {roc_auc_score(y_test, ensemble_scores):.4f}")

# ── Persist artefacts ──────────────────────────────────────────────────────
joblib.dump(xgb_model, "models/xgb.pkl")
joblib.dump(iso_model, "models/iso.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModels and scaler saved to models/")
