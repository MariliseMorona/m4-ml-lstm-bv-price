"""Configuração central do projeto.

Carrega variáveis de ambiente do arquivo `.env` e expõe caminhos,
hiperparâmetros e utilitários de I/O para metadados do modelo.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

SYMBOL: str = os.getenv("SYMBOL", "DIS")
START_DATE: str = os.getenv("START_DATE", "2018-01-01")
END_DATE: str = os.getenv("END_DATE", "2024-07-20")

LOOKBACK: int = int(os.getenv("LOOKBACK", "60"))
LSTM_UNITS: int = int(os.getenv("LSTM_UNITS", "50"))
DROPOUT: float = float(os.getenv("DROPOUT", "0.2"))
EPOCHS: int = int(os.getenv("EPOCHS", "100"))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.001"))

TRAIN_RATIO: float = float(os.getenv("TRAIN_RATIO", "0.70"))
VAL_RATIO: float = float(os.getenv("VAL_RATIO", "0.15"))
TEST_RATIO: float = float(os.getenv("TEST_RATIO", "0.15"))

API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))


def ensure_dirs() -> None:
    """Cria diretórios de dados, modelos e relatórios se não existirem."""
    for path in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def raw_csv_path(symbol: str | None = None) -> Path:
    """Retorna o caminho do CSV bruto baixado do Yahoo Finance."""
    sym = symbol or SYMBOL
    return DATA_RAW_DIR / f"{sym}_historical.csv"


def processed_npz_path(symbol: str | None = None) -> Path:
    """Retorna o caminho do arquivo NPZ com sequências treino/val/teste."""
    sym = symbol or SYMBOL
    return DATA_PROCESSED_DIR / f"{sym}_sequences.npz"


def model_path(symbol: str | None = None) -> Path:
    """Retorna o caminho do checkpoint PyTorch do modelo LSTM."""
    sym = symbol or SYMBOL
    return MODELS_DIR / f"lstm_{sym}.pt"


def scaler_path(symbol: str | None = None) -> Path:
    """Retorna o caminho do MinMaxScaler serializado com joblib."""
    sym = symbol or SYMBOL
    return MODELS_DIR / f"scaler_{sym}.pkl"


def metadata_path(symbol: str | None = None) -> Path:
    """Retorna o caminho do JSON com metadados e métricas do treino."""
    sym = symbol or SYMBOL
    return MODELS_DIR / f"metadata_{sym}.json"


def save_metadata(data: dict, symbol: str | None = None) -> Path:
    """Salva metadados do modelo (hiperparâmetros, métricas, datas) em JSON."""
    path = metadata_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_metadata(symbol: str | None = None) -> dict:
    """Carrega metadados do disco; retorna dict vazio se o arquivo não existir."""
    path = metadata_path(symbol)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)
