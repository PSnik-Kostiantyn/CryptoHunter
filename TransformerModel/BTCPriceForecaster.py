import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


class BTCPriceForecaster:
    def __init__(self, sequence_length=120):
        self.data_path = None
        self.model_path = None
        self.sequence_length = sequence_length
        self.features = ["Open", "High", "Low", "Close", "Volume", "Quote asset volume",
                         "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume",
                         "Score", "SMA_20", "RSI_14", "RSI_7", "MACD_14"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_and_prepare_data(self, data_path = "../datasets/BTC_ready.csv"):
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)
        self.data["Close time"] = pd.to_datetime(self.data["Close time"])
        self.data = self.data.sort_values("Close time")

        X = self.data[self.features].values
        y = self.data["Close"].values.reshape(-1, 1)

        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        self.X_scaled = X_scaled
        self.y_scaled = y_scaled

        self.X_seq, self.y_seq = self.create_sequences(X_scaled, y_scaled)

        test_size = int(len(self.X_seq) * 0.05)
        self.X_train = self.X_seq[:-test_size]
        self.y_train = self.y_seq[:-test_size]
        self.X_test = self.X_seq[-test_size:]
        self.y_test = self.y_seq[-test_size:]

        self.train_loader = DataLoader(TimeSeriesDataset(self.X_train, self.y_train), batch_size=32, shuffle=True)
        self.test_loader = DataLoader(TimeSeriesDataset(self.X_test, self.y_test), batch_size=32)

    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def build_or_load_model(self, model_path="../models/transformer_model.pth"):
        self.model_path = model_path
        self.model = TimeSeriesTransformer(input_dim=self.X_train.shape[2]).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print("Model loaded from disk.")
        else:
            print("Training model...")
            for epoch in range(35):
                self.model.train()
                total_loss = 0
                for X_batch, y_batch in self.train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(X_batch).squeeze()
                    loss = self.criterion(output, y_batch.squeeze())
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                print(f"Epoch {epoch + 1}/30, Loss: {total_loss / len(self.train_loader):.6f}")
            torch.save(self.model.state_dict(), self.model_path)
            print("Model trained and saved.")

    def evaluate_model(self):
        print("\nEvaluating on test set...")
        self.model.eval()
        predictions = []
        actual = []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                output = self.model(X_batch).cpu().detach().numpy().reshape(-1)
                predictions.extend(output)
                actual.extend(y_batch.cpu().numpy().reshape(-1))

        predictions = self.scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))
        actual = self.scaler_y.inverse_transform(np.array(actual).reshape(-1, 1))

        mae = mean_absolute_error(actual, predictions)
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        rel_error = np.mean(np.abs(predictions - actual) / actual) * 100

        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"Mean Relative Error: {rel_error:.2f}%")

        plt.figure(figsize=(12, 6))
        plt.plot(actual, label="Actual", color='orange')
        plt.plot(predictions, label="Predicted", color='blue')
        plt.title("Model Evaluation on Test Set")
        plt.xlabel("Samples")
        plt.ylabel("BTC Close Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def forecast(self, steps=6):
        self.data = pd.read_csv(self.data_path)

        if np.issubdtype(self.data["Close time"].dtype, np.number):
            self.data["Close time"] = pd.to_datetime(self.data["Close time"], unit='s')
        else:
            self.data["Close time"] = pd.to_datetime(self.data["Close time"])

        self.data = self.data.sort_values("Close time")

        X = self.data[self.features].values

        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        X_scaled = self.scaler_X.fit_transform(X)

        last_sequence = X_scaled[-self.sequence_length:]
        last_sequence = last_sequence.copy()
        input_seq = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        close_index = self.features.index("Close")
        current_time = self.data["Close time"].iloc[-1]
        self.model.eval()
        predictions = []

        for step in range(steps):
            with torch.no_grad():
                pred_scaled = self.model(input_seq).item()

            pred_unscaled = float(self.scaler_y.inverse_transform([[pred_scaled]])[0][0])
            future_time = current_time + pd.Timedelta(hours=step + 1)
            timestamp = int(future_time.timestamp())

            predictions.append({
                "timestamp": timestamp,
                "close": pred_unscaled
            })

            last_features = last_sequence[-1].copy()
            last_features[close_index] = pred_scaled

            new_sequence = np.roll(last_sequence, -1, axis=0)
            new_sequence[-1] = last_features

            last_sequence = new_sequence
            input_seq = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        return predictions

    def train_model_on_new_data(self, from_timestamp, csv_path="../datasets/BTC_ready.csv"):
        print("Перевірка оновлення CSV-файлу...")

        latest_data = pd.read_csv(csv_path)

        if "Close time" not in latest_data.columns:
            print("Помилка: у CSV відсутня колонка 'Close time'")
            return

        latest_data["Close time"] = latest_data["Close time"].astype(int)

        last_ts_in_csv = latest_data["Close time"].max()

        if from_timestamp >= last_ts_in_csv:
            print(f"CSV не оновився. from_timestamp = {from_timestamp}, останній в CSV = {last_ts_in_csv}")
            return

        self.data = latest_data.copy()

        print(f"CSV оновився. Нові дані донавчаються з {from_timestamp} до {last_ts_in_csv}")

        df_full = self.data[self.data["Close time"] <= last_ts_in_csv]
        new_data_start_idx = df_full[df_full["Close time"] > from_timestamp].index.min()

        if pd.isna(new_data_start_idx):
            print("Немає нових рядків у CSV.")
            return

        start_idx = max(0, new_data_start_idx - self.sequence_length)
        df_for_sequences = df_full.iloc[start_idx:]

        from_dt = pd.to_datetime(from_timestamp, unit='s')
        last_dt = pd.to_datetime(last_ts_in_csv, unit='s')
        print(
            f"Донавчання на даних з {from_dt} до {last_dt} ({len(df_for_sequences)} записів для побудови послідовностей)")

        X = df_for_sequences[self.features].values
        y = df_for_sequences["Close"].values.reshape(-1, 1)

        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.scaler_y.transform(y)

        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)

        close_times = df_for_sequences["Close time"].values
        target_times = close_times[self.sequence_length:]

        mask = target_times > from_timestamp
        X_seq = X_seq[mask]
        y_seq = y_seq[mask]

        if len(X_seq) == 0:
            print("Недостатньо даних для формування нових послідовностей.")
            return

        X_new, y_new = X_seq[-500:], y_seq[-500:]
        new_loader = DataLoader(TimeSeriesDataset(X_new, y_new), batch_size=32, shuffle=True)

        if not hasattr(self, 'model'):
            self.build_or_load_model()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.train_model(self.model, new_loader, optimizer, epochs=5)
        print("Донавчання завершено.")

    def train_model(self, model, loader, optimizer, epochs=5):
        criterion = torch.nn.MSELoss()
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                output = model(X_batch).squeeze()
                loss = criterion(output, y_batch.squeeze())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.6f}")
        torch.save(model.state_dict(), self.model_path)
        print("Модель дотренована та збережена.")
