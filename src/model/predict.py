"""Inferência em produção com o modelo LSTM treinado."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch

from src.config import LOOKBACK, load_metadata, model_path, scaler_path
from src.model.io import get_device, load_model


class StockPredictor:
    """Encapsula modelo, scaler e metadados para servir previsões.

    Attributes:
        symbol: Ticker associado ao modelo.
        lookback: Janela mínima de preços históricos.
        device: Dispositivo PyTorch (cpu/cuda/mps).
        model: Rede LSTM em modo eval.
        scaler: MinMaxScaler do pré-processamento.
    """

    def __init__(self, symbol: str, models_dir: Path | None = None):
        """Carrega artefatos do disco.

        Args:
            symbol: Ticker (ex.: ``DIS``).
            models_dir: Diretório alternativo de modelos (opcional).

        Raises:
            FileNotFoundError: Se modelo ou scaler não existirem.
        """
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
        """Prevê um ou mais fechamentos futuros.

        Para ``steps > 1``, cada previsão alimenta a janela seguinte
        (previsão recursiva).

        Args:
            prices: Histórico de fechamentos (mínimo ``lookback`` valores).
            steps: Quantidade de dias à frente.

        Returns:
            Lista com preços previstos em escala original.

        Raises:
            ValueError: Se ``steps < 1``, preços insuficientes, NaN ou ≤ 0.
        """
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
    """Atalho para instanciar ``StockPredictor``."""
    return StockPredictor(symbol)
