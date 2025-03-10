import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("datasets/BTC_ready.csv")

data["Close time"] = pd.to_datetime(data["Close time"])
data = data.sort_values("Close time")

features = ["Open", "High", "Low", "Close", "Volume", "Quote asset volume",
            "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume",
            "Score", "SMA_20", "RSI_14", "RSI_7", "MACD_14"]

data["Close_next"] = data["Close"].shift(-1)

data.dropna(inplace=True)

X = data[features]
y = data["Close_next"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=False)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")

plt.figure(figsize=(12, 6))
plt.plot(data["Close time"].iloc[-len(y_test):], predictions, label="Predictions", color='red', linewidth=2)
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("Our Stock Predictions")
plt.legend()
plt.grid(True)
plt.show()