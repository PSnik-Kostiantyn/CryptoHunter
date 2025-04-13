import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset
import os

data = pd.read_csv("../datasets/BTC_ready.csv")
data["Close time"] = pd.to_datetime(data["Close time"])
data = data.sort_values("Close time")

features = ["Open", "High", "Low", "Close", "Volume", "Quote asset volume",
            "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume",
            "Score", "SMA_20", "RSI_14", "RSI_7", "MACD_14"]

X = data[features].values
y = data["Close"].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

sequence_length = 120

def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)

X_train = X_seq
y_train = y_seq

test_size = int(len(X_seq) * 0.2)
X_test = X_seq[-test_size:]
y_test = y_seq[-test_size:]


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=32)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, max_len=500):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :])

def load_or_train_model(model, train_loader, test_loader, model_path, criterion, optimizer, scaler_y):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model loaded from disk.")
    else:
        print("Training model...")
        for epoch in range(30):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                output = model(X_batch).squeeze()
                loss = criterion(output, y_batch.squeeze())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/20, Loss: {total_loss / len(train_loader):.6f}")
        torch.save(model.state_dict(), model_path)
        print("Model trained and saved.")
        evaluate_model(model, test_loader, scaler_y)
    return model

def evaluate_model(model, test_loader, scaler_y):
    print("\nEvaluating on test set...")
    model.eval()
    predictions = []
    actuals = []

    if len(test_loader.dataset) == 0:
        print("Попередження: тестовий набір порожній. Оцінка моделі пропущена.")
        return

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch).squeeze().cpu().numpy()
            predictions.extend(output)
            actuals.extend(y_batch.squeeze().numpy())

    predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))
    actuals = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1))

    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    rel_error = np.mean(np.abs(predictions - actuals) / actuals) * 100

    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Mean Relative Error: {rel_error:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label="Actual", color='orange')
    plt.plot(predictions, label="Predicted", color='blue')
    plt.title("Model Evaluation on Test Set")
    plt.xlabel("Samples")
    plt.ylabel("BTC Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def train_model(model, train_loader, optimizer, epochs=5):
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.6f}")

    torch.save(model.state_dict(), model_path)
    print("Модель дотренована та збережена.")


def train_model_on_new_data(df):
    print("Підготовка нових даних для донавчання...")

    features = ["Open", "High", "Low", "Close", "Volume", "Quote asset volume",
                "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume",
                "Score", "SMA_20", "RSI_14", "RSI_7", "MACD_14"]

    df["Close time"] = pd.to_datetime(df["Close time"])
    df = df.sort_values("Close time")

    X = df[features].values
    y = df["Close"].values.reshape(-1, 1)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    sequence_length = 120
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)

    N = 500
    X_new, y_new = X_seq[-N:], y_seq[-N:]

    new_loader = DataLoader(TimeSeriesDataset(X_new, y_new), batch_size=32, shuffle=True)

    model = TimeSeriesTransformer(input_dim=X_new.shape[2]).to(device)
    model_path = "../models/transformer_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Модель завантажена для донавчання.")
    else:
        print("Не знайдено існуючої моделі, створюю нову...")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_model(model, new_loader, optimizer, epochs=5)

    print("Донавчання завершено.")


def forecast(model, last_sequence, steps, last_timestamp):
    model.eval()
    predictions = []
    input_seq = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    current_time = pd.to_datetime(last_timestamp)

    for step in range(steps):
        with torch.no_grad():
            pred = model(input_seq).item()
        timestamp = current_time + pd.Timedelta(hours=step + 1)
        predictions.append({
            "timestamp": int(timestamp.timestamp()),
            "close": float(scaler_y.inverse_transform([[pred]])[0][0])
        })

        new_input = np.roll(last_sequence, -1, axis=0)
        new_input[-1] = np.append(new_input[-1, :-1], pred)
        last_sequence = new_input
        input_seq = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)

    return predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesTransformer(input_dim=X_train.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model_path = "../models/transformer_model.pth"

model = load_or_train_model(model, train_loader, test_loader, model_path, criterion, optimizer, scaler_y)

last_sequence = X_scaled[-sequence_length:]
last_timestamp = data["Close time"].iloc[-1]
pred_future = forecast(model, last_sequence, steps=12, last_timestamp=last_timestamp)

for item in pred_future:
    print(item)

timestamps = [pd.to_datetime(item["timestamp"], unit='s') for item in pred_future]
prices = [item["close"] for item in pred_future]

plt.figure(figsize=(10, 5))
plt.plot(timestamps, prices, marker='o', linestyle='-', color='b', label='Прогноз ціни BTC')
plt.xlabel('Дата і час')
plt.ylabel('Ціна закриття BTC')
plt.title(f'Прогноз ціни BTC на {len(pred_future)} годин вперед')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
