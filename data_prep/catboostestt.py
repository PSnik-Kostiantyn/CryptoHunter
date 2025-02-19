import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv("../datasets/technical_indicators.csv")

data["Timestamp"] = pd.to_datetime(data["Timestamp"], unit='s')

data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

data = data[:-1]

features = ["Open", "High", "Low", "Close", "SMA_20", "RSI_14", "RSI_7", "MACD_14"]
X = data[features]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

model = CatBoostClassifier(iterations=5000, depth=6, learning_rate=0.05, loss_function='Logloss', verbose=100)

model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, use_best_model=True)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Точність моделі: {accuracy:.4f}")

joblib.dump(model, "catboost_bitcoin_model.pkl")
print("Модель збережено у catboost_bitcoin_model.pkl")
