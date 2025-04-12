def train_model(updated_data):
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from torch.utils.data import DataLoader, Dataset
    import matplotlib.pyplot as plt
    import os

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
            self.positional_encoding = PositionalEncoding(d_model, max_len)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                       dim_feedforward=dim_feedforward, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x):
            x = self.input_projection(x)
            x = self.positional_encoding(x)
            x = self.transformer_encoder(x)
            return self.fc(x[:, -1, :])

    def create_sequences(X, y, seq_length):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i + seq_length])
            y_seq.append(y[i + seq_length])
        return np.array(X_seq), np.array(y_seq)

    features = ["Open", "High", "Low", "Close", "Volume", "Quote asset volume",
                "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume",
                "Score", "SMA_20", "RSI_14", "RSI_7", "MACD_14"]

    updated_data["Close time"] = pd.to_datetime(updated_data["Close time"])
    updated_data = updated_data.sort_values("Close time")

    X = updated_data[features].values
    y = updated_data["Close"].values.reshape(-1, 1)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    sequence_length = 120
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)

    train_loader = DataLoader(TimeSeriesDataset(X_seq, y_seq), batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer(input_dim=X_seq.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model_path = "../models/transformer_model.pth"

    print("Тренування моделі на оновлених даних...")

    for epoch in range(10):
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
        print(f"Epoch {epoch + 1}/10, Loss: {total_loss / len(train_loader):.6f}")

    torch.save(model.state_dict(), model_path)
    print("Модель збережено за адресою:", model_path)
