import pandas as pd
import joblib
import xgboost as xgb

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("creditcard.csv")

# Scale Amount
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# XGBoost
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight
)

xgb_model.fit(X_train, y_train)

# Isolation Forest
iso_model = IsolationForest(contamination=0.002)
iso_model.fit(X_train)

joblib.dump(xgb_model, "models/xgb.pkl")
joblib.dump(iso_model, "models/iso.pkl")

print("Models trained and saved")
