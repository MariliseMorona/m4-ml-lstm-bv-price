"""Avaliação do modelo LSTM no conjunto de teste."""

from __future__ import annotations

import json
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import REPORTS_DIR, SYMBOL, ensure_dirs, load_metadata, model_path, processed_npz_path, save_metadata, scaler_path
from src.model.io import get_device, load_model
from src.model.metrics import compute_all


def evaluate(symbol: str | None = None) -> dict:
    """Calcula MAE, RMSE e MAPE no teste e gera gráfico de comparação.

    Args:
        symbol: Ticker (padrão: ``SYMBOL``).

    Returns:
        Dicionário com métricas ``mae``, ``rmse`` e ``mape``.
    """
    sym = symbol or SYMBOL
    ensure_dirs()
    device = get_device()

    data = np.load(processed_npz_path(sym), allow_pickle=True)
    model = load_model(model_path(sym), device)
    scaler = joblib.load(scaler_path(sym))

    X_test = torch.tensor(data["X_test"], dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred_scaled = model(X_test).cpu().numpy().flatten()

    y_true_scaled = data["y_test"].flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

    metrics = compute_all(y_true, y_pred)
    print(f"Test metrics for {sym}:")
    for name, value in metrics.items():
        print(f"  {name.upper()}: {value:.4f}")

    meta = load_metadata(sym)
    meta["test_metrics"] = metrics
    save_metadata(meta, sym)

    report_path = REPORTS_DIR / f"{sym}_test_metrics.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {report_path}")

    _plot_predictions(sym, y_true, y_pred)
    return metrics


def _plot_predictions(symbol: str, y_true: np.ndarray, y_pred: np.ndarray, n: int = 120) -> None:
    """Salva gráfico PNG comparando série real e prevista (últimos n pontos)."""
    n = min(n, len(y_true))
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[-n:], label="Real", linewidth=2)
    plt.plot(y_pred[-n:], label="Previsto", linewidth=2, alpha=0.85)
    plt.title(f"Previsão vs Real — {symbol} (últimos {n} dias de teste)")
    plt.xlabel("Índice")
    plt.ylabel("Preço de fechamento")
    plt.legend()
    plt.tight_layout()
    out = REPORTS_DIR / f"{symbol}_predictions.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved to {out}")


def main() -> None:
    """Executa a avaliação via linha de comando."""
    try:
        evaluate()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
