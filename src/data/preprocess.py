"""Pré-processamento de séries temporais para treino da LSTM."""

from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    LOOKBACK,
    SYMBOL,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    ensure_dirs,
    processed_npz_path,
    raw_csv_path,
    scaler_path,
)


def load_raw(symbol: str | None = None) -> pd.DataFrame:
    """Carrega o CSV bruto gerado por ``download``.

    Args:
        symbol: Ticker (padrão: ``SYMBOL`` do .env).

    Raises:
        FileNotFoundError: Se o CSV não existir.
    """
    path = raw_csv_path(symbol)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run: python -m src.data.download"
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def build_sequences(prices: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Monta janelas deslizantes para aprendizado supervisionado.

    Args:
        prices: Série 1D de preços já normalizados.
        lookback: Quantidade de dias de entrada.

    Returns:
        Tupla ``(X, y)`` onde cada ``X[i]`` tem shape ``(lookback,)`` e
        ``y[i]`` é o preço do dia seguinte.
    """
    X, y = [], []
    for i in range(lookback, len(prices)):
        X.append(prices[i - lookback : i])
        y.append(prices[i])
    return np.array(X), np.array(y)


def temporal_split(
    n_samples: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[slice, slice, slice]:
    """Divide amostras em ordem cronológica (sem embaralhar).

    Args:
        n_samples: Número total de sequências.
        train_ratio: Proporção de treino (ex.: 0.70).
        val_ratio: Proporção de validação.
        test_ratio: Proporção de teste.

    Returns:
        Slices ``(train, val, test)`` para indexar arrays.

    Raises:
        ValueError: Se as proporções não somarem 1.0.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n_samples)


def preprocess(symbol: str | None = None, lookback: int | None = None) -> dict:
    """Pipeline completo: normalização, sequências e split temporal.

    O ``MinMaxScaler`` é ajustado **apenas** na porção de treino dos preços
    brutos, evitando vazamento de informação para validação e teste.

    Args:
        symbol: Ticker (padrão: ``SYMBOL``).
        lookback: Tamanho da janela (padrão: ``LOOKBACK``).

    Returns:
        Dicionário com arrays ``X_train``, ``y_train``, ``X_val``, ``y_val``,
        ``X_test``, ``y_test``, além de ``dates``, ``lookback`` e ``symbol``.
    """
    sym = symbol or SYMBOL
    lb = lookback or LOOKBACK

    ensure_dirs()
    df = load_raw(sym)

    if "Close" not in df.columns:
        raise ValueError("Expected 'Close' column in raw data.")

    close = df["Close"].astype(float).values.reshape(-1, 1)
    close = close[~np.isnan(close).flatten()]

    n = len(close)
    train_end = int(n * TRAIN_RATIO)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(close[:train_end])
    scaled = scaler.transform(close).flatten()

    X, y = build_sequences(scaled, lb)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    train_sl, val_sl, test_sl = temporal_split(
        len(X), TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )

    splits = {
        "X_train": X[train_sl],
        "y_train": y[train_sl],
        "X_val": X[val_sl],
        "y_val": y[val_sl],
        "X_test": X[test_sl],
        "y_test": y[test_sl],
        "dates": df["Date"].iloc[lb:].astype(str).tolist(),
        "lookback": lb,
        "symbol": sym,
    }

    np.savez_compressed(processed_npz_path(sym), **splits)
    joblib.dump(scaler, scaler_path(sym))

    print(f"Sequences: train={len(splits['X_train'])}, val={len(splits['X_val'])}, test={len(splits['X_test'])}")
    print(f"Saved to {processed_npz_path(sym)}")
    print(f"Scaler saved to {scaler_path(sym)}")
    return splits


def main() -> None:
    """Executa o pré-processamento via linha de comando."""
    try:
        preprocess()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
