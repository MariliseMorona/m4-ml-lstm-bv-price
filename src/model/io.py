"""Save and load PyTorch model checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from src.config import DROPOUT, LSTM_UNITS
from src.model.lstm import LSTMStockPredictor


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    path: Path,
    model: LSTMStockPredictor,
    lookback: int,
    lstm_units: int = LSTM_UNITS,
    dropout: float = DROPOUT,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "lookback": lookback,
            "lstm_units": lstm_units,
            "dropout": dropout,
            "framework": "pytorch",
        },
        path,
    )


def load_model(path: Path, device: torch.device | None = None) -> LSTMStockPredictor:
    dev = device or get_device()
    checkpoint = torch.load(path, map_location=dev, weights_only=False)

    lookback = int(checkpoint["lookback"])
    model = LSTMStockPredictor(
        lookback=lookback,
        units=int(checkpoint.get("lstm_units", LSTM_UNITS)),
        dropout=float(checkpoint.get("dropout", DROPOUT)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(dev)
    model.eval()
    return model
