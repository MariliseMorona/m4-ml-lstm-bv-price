"""Inference utilities for the trained LSTM model."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch

from src.config import LOOKBACK, load_metadata, model_path, scaler_path
from src.model.io import get_device, load_model


class StockPredictor:
    def __init__(self, symbol: str, models_dir: Path | None = None):
        self.symbol = symbol
        self.metadata = load_metadata(symbol)
        self.lookback = int(self.metadata.get("lookback", LOOKBACK))
        self.device = get_device()

        mpath = model_path(symbol) if models_dir is None else models_dir / f"lstm_{symbol}.pt"
        spath = scaler_path(symbol) if models_dir is None else models_dir / f"scaler_{symbol}.pkl"

        if not mpath.exists():
            raise FileNotFoundError(f"Model not found: {mpath}. Run training first.")
        if not spath.exists():
            raise FileNotFoundError(f"Scaler not found: {spath}. Run preprocessing first.")

        self.model = load_model(mpath, self.device)
        self.scaler = joblib.load(spath)

    def predict_next(self, prices: list[float] | np.ndarray, steps: int = 1) -> list[float]:
        if steps < 1:
            raise ValueError("steps must be >= 1")

        seq = np.array(prices, dtype=float).flatten()
        if len(seq) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} prices, got {len(seq)}")
        if np.any(np.isnan(seq)) or np.any(seq <= 0):
            raise ValueError("Prices must be positive numbers without NaN.")

        window = seq[-self.lookback :].reshape(-1, 1)
        scaled = self.scaler.transform(window).flatten()
        predictions: list[float] = []

        with torch.no_grad():
            for _ in range(steps):
                X = torch.tensor(
                    scaled[-self.lookback :],
                    dtype=torch.float32,
                    device=self.device,
                ).reshape(1, self.lookback, 1)
                pred_scaled = self.model(X).item()
                pred_price = float(self.scaler.inverse_transform([[pred_scaled]])[0, 0])
                predictions.append(pred_price)
                scaled = np.append(scaled, pred_scaled)

        return predictions


def load_predictor(symbol: str) -> StockPredictor:
    return StockPredictor(symbol)
