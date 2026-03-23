import joblib
import numpy as np

xgb_model = joblib.load("models/xgb.pkl")
iso_model = joblib.load("models/iso.pkl")

def predict_score(data):

    X = np.array([list(data.values())])

    xgb_score = xgb_model.predict_proba(X)[0][1]

    iso_pred = iso_model.predict(X)[0]
    iso_score = 1 if iso_pred == -1 else 0

    final_score = (0.7 * xgb_score) + (0.3 * iso_score)

    return float(final_score)
