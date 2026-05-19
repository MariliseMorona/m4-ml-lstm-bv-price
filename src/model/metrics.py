"""Métricas de avaliação para previsão de preços de ações."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Erro absoluto médio (Mean Absolute Error).

    Args:
        y_true: Valores reais.
        y_pred: Valores previstos.

    Returns:
        MAE em unidade do preço (ex.: US$).
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Raiz do erro quadrático médio (Root Mean Square Error).

    Penaliza mais erros grandes que o MAE.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Erro percentual absoluto médio (Mean Absolute Percentage Error).

    Args:
        epsilon: Evita divisão por zero quando o preço é muito baixo.

    Returns:
        MAPE em porcentagem (0–100).
    """
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula MAE, RMSE e MAPE de uma vez.

    Returns:
        Dicionário com chaves ``mae``, ``rmse`` e ``mape``.
    """
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }
