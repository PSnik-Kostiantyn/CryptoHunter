import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset
import os

class BTCPriceForecaster:
    def __init__(self, data_path, model_path="../models/transformer_model.pth", sequence_length=120):
        self.data_path = data_path
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.features = [
            "Open", "High", "Low", "Close", "Volume", "Quote asset volume",
            "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume",
            "Score", "SMA_20", "RSI_14", "RSI_7", "MACD_14"
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None
        self.data = None
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None

    def load_and_prepare_data(self):
        data = pd.read_csv(self.data_path)
        data["Close time"] = pd.to_datetime(data["Close time"])
        data = data.sort_values("Close time")
        self.data = data

        X = data[self.features].values
        y = data["Close"].values.reshape(-1, 1)
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)

        test_size = int(len(X_seq) * 0.2)
        self.X_train, self.y_train = X_seq, y_seq
        self.X_test, self.y_test = X_seq[-test_size:], y_seq[-test_size:]

    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)

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
            return x + self.pe[:, :x.size(1)]

    class TimeSeriesTransformer(nn.Module):
        def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, max_len=500):
            super().__init__()
            self.input_projection = nn.Linear(input_dim, d_model)
            self.positional_encoding = BTCPriceForecaster.PositionalEncoding(d_model, max_len)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                       dim_feedforward=dim_feedforward, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x):
            x = self.input_projection(x)
            x = self.positional_encoding(x)
            x = self.transformer_encoder(x)
            return self.fc(x[:, -1, :])

    def build_or_load_model(self):
        self.model = self.TimeSeriesTransformer(input_dim=self.X_train.shape[2]).to(self.device)
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print("Model loaded from disk.")
        else:
            print("Training model from scratch...")
            self.train_model()
            torch.save(self.model.state_dict(), self.model_path)

    def train_model(self, epochs=2, lr=1e-3):
        train_loader = DataLoader(self.TimeSeriesDataset(self.X_train, self.y_train), batch_size=32, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                output = self.model(X_batch).squeeze()
                loss = criterion(output, y_batch.squeeze())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.6f}")

    def evaluate_model(self):
        test_loader = DataLoader(self.TimeSeriesDataset(self.X_test, self.y_test), batch_size=32)
        self.model.eval()
        predictions, actuals = [], []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                output = self.model(X_batch).squeeze().cpu().numpy()
                predictions.extend(output)
                actuals.extend(y_batch.squeeze().numpy())

        predictions = self.scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))
        actuals = self.scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1))

        mae = mean_absolute_error(actuals, predictions)
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        rel_error = np.mean(np.abs(predictions - actuals) / actuals) * 100

        print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, Mean Relative Error: {rel_error:.2f}%")

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

    def forecast(self, steps=12):
        last_sequence = self.scaler_X.transform(self.data[self.features].values)[-self.sequence_length:]
        last_timestamp = self.data["Close time"].iloc[-1]
        self.model.eval()

        predictions = []
        input_seq = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        current_time = pd.to_datetime(last_timestamp)

        for step in range(steps):
            with torch.no_grad():
                pred = self.model(input_seq).item()
            timestamp = current_time + pd.Timedelta(hours=step + 1)
            predictions.append({
                "timestamp": int(timestamp.timestamp()),
                "close": float(self.scaler_y.inverse_transform([[pred]])[0][0])
            })

            new_input = np.roll(last_sequence, -1, axis=0)
            new_input[-1] = np.append(new_input[-1, :-1], pred)
            last_sequence = new_input
            input_seq = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        return predictions


