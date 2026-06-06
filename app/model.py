from pathlib import Path

import joblib
import numpy as np

_MODELS_DIR = Path(__file__).parent.parent / "models"

xgb_model = joblib.load(_MODELS_DIR / "xgb.pkl")
iso_model = joblib.load(_MODELS_DIR / "iso.pkl")
scaler = joblib.load(_MODELS_DIR / "scaler.pkl")


def predict_score(data: dict) -> float:
    values = list(data.values())
    X = np.array([values])

    # Apply the same Amount scaling used during training (last feature column)
    X[:, -1] = scaler.transform(X[:, -1].reshape(-1, 1)).flatten()

    xgb_score: float = xgb_model.predict_proba(X)[0][1]

    iso_pred: int = iso_model.predict(X)[0]
    iso_score: float = 1.0 if iso_pred == -1 else 0.0

    final_score = (0.7 * xgb_score) + (0.3 * iso_score)
    return float(final_score)
