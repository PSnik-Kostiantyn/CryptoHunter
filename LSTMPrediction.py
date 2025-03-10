import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv("datasets/BTC_ready.csv")
data["Close time"] = pd.to_datetime(data["Close time"])
data = data.sort_values("Close time")

features = ["Open", "High", "Low", "Close", "Volume", "Quote asset volume",
            "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume",
            "Score", "SMA_20", "RSI_14", "RSI_7", "MACD_14"]

data["Close_next"] = data["Close"].shift(-1)

data.dropna(inplace=True)

X = data[features].values
y = data["Close_next"].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

sequence_length = 50

def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)

train_size = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(sequence_length, X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

mae = mean_absolute_error(y_test_actual, y_pred)
mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

plt.figure(figsize=(12, 6))
plt.plot(data["Close time"].iloc[-len(y_test_actual):], y_test_actual, label="Test (Actual)", color='orange', linewidth=2)
plt.plot(data["Close time"].iloc[-len(y_pred):], y_pred, label="Predictions", color='red', linewidth=2)
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("LSTM Stock Predictions")
plt.legend()
plt.grid(True)
plt.show()
