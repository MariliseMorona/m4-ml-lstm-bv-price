"""LSTM model definition (PyTorch)."""

from __future__ import annotations

import torch
from torch import nn

from src.config import DROPOUT, LSTM_UNITS


class LSTMStockPredictor(nn.Module):
    def __init__(
        self,
        lookback: int,
        units: int | None = None,
        dropout: float | None = None,
    ):
        super().__init__()
        u = units or LSTM_UNITS
        d = dropout if dropout is not None else DROPOUT

        self.lookback = lookback
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=u, batch_first=True)
        self.dropout1 = nn.Dropout(d)
        self.lstm2 = nn.LSTM(input_size=u, hidden_size=u, batch_first=True)
        self.dropout2 = nn.Dropout(d)
        self.fc = nn.Linear(u, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])
        return self.fc(out)


def build_lstm_model(
    lookback: int,
    units: int | None = None,
    dropout: float | None = None,
) -> LSTMStockPredictor:
    return LSTMStockPredictor(lookback=lookback, units=units, dropout=dropout)
