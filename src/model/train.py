"""Treinamento do modelo LSTM com early stopping e checkpoint."""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    BATCH_SIZE,
    DROPOUT,
    EPOCHS,
    END_DATE,
    LEARNING_RATE,
    LOOKBACK,
    LSTM_UNITS,
    START_DATE,
    SYMBOL,
    ensure_dirs,
    model_path,
    processed_npz_path,
    save_metadata,
)
from src.model.io import get_device, save_checkpoint
from src.model.lstm import build_lstm_model


def load_processed(symbol: str | None = None) -> dict:
    """Carrega arrays NumPy do arquivo NPZ processado.

    Raises:
        FileNotFoundError: Se o pré-processamento não foi executado.
    """
    path = processed_npz_path(symbol)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. Run: python -m src.data.preprocess"
        )
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    """Converte arrays em DataLoader PyTorch."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=shuffle)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    """Executa uma época de treino ou validação.

    Returns:
        Loss média da época.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    losses: list[float] = []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            if is_train:
                loss.backward()
                optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


def train(symbol: str | None = None) -> dict:
    """Treina a LSTM com Adam, MSE e early stopping.

    Salva o melhor checkpoint (menor val_loss) e metadados em JSON.

    Args:
        symbol: Ticker (padrão: ``SYMBOL``).

    Returns:
        Dicionário com ``history`` (loss por época) e ``metadata``.
    """
    sym = symbol or SYMBOL
    ensure_dirs()

    data = load_processed(sym)
    lookback = int(data["lookback"])
    device = get_device()
    print(f"Using device: {device}")

    model = build_lstm_model(lookback=lookback).to(device)
    print(model)

    train_loader = _make_loader(data["X_train"], data["y_train"], shuffle=True)
    val_loader = _make_loader(data["X_val"], data["y_val"], shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 15
    history: dict[str, list[float]] = {"loss": [], "val_loss": []}
    mpath = model_path(sym)

    for epoch in range(1, EPOCHS + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _run_epoch(model, val_loader, criterion, None, device)
        scheduler.step(val_loss)

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch}/{EPOCHS} — loss: {train_loss:.6f} — val_loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(mpath, model, lookback, LSTM_UNITS, DROPOUT)
            print(f"  Saved best model (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    meta = {
        "symbol": sym,
        "lookback": lookback,
        "lstm_units": LSTM_UNITS,
        "framework": "pytorch",
        "device": str(device),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "epochs_run": len(history["loss"]),
        "final_train_loss": history["loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "best_val_loss": best_val_loss,
    }
    save_metadata(meta, sym)
    print(f"Model saved to {mpath}")
    return {"history": history, "metadata": meta}


def main() -> None:
    """Executa o treinamento via linha de comando."""
    try:
        train()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
