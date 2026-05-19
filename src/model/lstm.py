"""Definição da rede neural LSTM (PyTorch) para previsão de preços."""

from __future__ import annotations

import torch
from torch import nn

from src.config import DROPOUT, LSTM_UNITS


class LSTMStockPredictor(nn.Module):
    """Rede LSTM empilhada para prever o próximo preço de fechamento.

    Arquitetura: LSTM → Dropout → LSTM → Dropout → Linear(1).

    Attributes:
        lookback: Tamanho da janela de entrada (dias).
    """

    def __init__(
        self,
        lookback: int,
        units: int | None = None,
        dropout: float | None = None,
    ):
        """Inicializa as camadas da rede.

        Args:
            lookback: Dias de histórico por amostra.
            units: Neurônios em cada camada LSTM (padrão: ``LSTM_UNITS``).
            dropout: Taxa de dropout (padrão: ``DROPOUT``).
        """
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
        """Forward pass.

        Args:
            x: Tensor de shape ``(batch, seq_len, 1)``.

        Returns:
            Previsão de shape ``(batch, 1)``.
        """
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
    """Cria uma instância do preditor LSTM com hiperparâmetros padrão.

    Args:
        lookback: Tamanho da janela temporal.
        units: Neurônios LSTM (opcional).
        dropout: Taxa de dropout (opcional).

    Returns:
        Modelo PyTorch não treinado.
    """
    return LSTMStockPredictor(lookback=lookback, units=units, dropout=dropout)
