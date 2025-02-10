import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("../datasets/technical_indicators.csv")

data = data.sort_values(by="Timestamp")

for lag in range(1, 6):
    data[f"Close_lag_{lag}"] = data["Close"].shift(lag)

data = data.dropna()

data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

data = data[:-1]

X = data.drop(columns=["Timestamp", "target"])
y = data["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Точність моделі: {accuracy:.4f}")

joblib.dump((model, scaler), "xgboost_model.pkl")
print("Модель збережено як 'xgboost_model.pkl'")

loaded_model, loaded_scaler = joblib.load("xgboost_model.pkl")
test_data = np.array([X_test[89]])
probability = loaded_model.predict_proba(test_data)[0, 1]
print(f"Ймовірність зростання ціни: {probability:.4f}")
