"""
Standalone model definitions for the Traffic Flow Prediction project.

Import from this module (not from the notebook/script) when you just want to load a
trained model and run inference — importing the notebook's .py export would re-execute
the entire training pipeline as a side effect, since it's a plain top-to-bottom script.

Usage:
    import torch, joblib
    from model_defs import GRUForecaster

    x_scaler = joblib.load("models/x_scaler.pkl")
    y_scaler = joblib.load("models/y_scaler.pkl")

    model = GRUForecaster(n_features=13, hidden_size=32, num_layers=1, horizon=1)
    model.load_state_dict(torch.load("models/gru_model.pt", map_location="cpu"))
    model.eval()

    # x: numpy array of shape (1, 12, 13) -- 12 lookback steps x 13 features, in the
    # exact FEATURE_COLS order used in the notebook (see README), scaled with x_scaler.
    with torch.no_grad():
        pred_scaled = model(torch.tensor(x, dtype=torch.float32)).numpy()
    pred_mph = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape)
"""

import numpy as np
import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=1, horizon=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUForecaster(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=1, horizon=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2, horizon=1, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.fc(x[:, -1, :])
