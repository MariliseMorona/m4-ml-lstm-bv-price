"""Coleta de preços históricos de ações via Yahoo Finance (yfinance)."""

from __future__ import annotations

import sys

import pandas as pd
import yfinance as yf

from src.config import END_DATE, START_DATE, SYMBOL, ensure_dirs, raw_csv_path


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Achata colunas MultiIndex retornadas pelo yfinance.

    Args:
        df: DataFrame retornado por ``yf.download``.

    Returns:
        DataFrame com colunas de primeiro nível (Close, Open, etc.).
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download(
    symbol: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Baixa histórico OHLCV e persiste em ``data/raw/{symbol}_historical.csv``.

    Args:
        symbol: Ticker Yahoo Finance (padrão: ``SYMBOL`` do .env).
        start: Data inicial ``YYYY-MM-DD``.
        end: Data final ``YYYY-MM-DD``.

    Returns:
        DataFrame com colunas Date, Open, High, Low, Close, Volume.

    Raises:
        ValueError: Se o yfinance não retornar dados para o ticker/período.
    """
    sym = symbol or SYMBOL
    start_date = start or START_DATE
    end_date = end or END_DATE

    ensure_dirs()
    print(f"Downloading {sym} from {start_date} to {end_date}...")

    df = yf.download(sym, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for symbol '{sym}'. Check ticker and date range.")

    df = flatten_columns(df)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"])

    out_path = raw_csv_path(sym)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    return df


def main() -> None:
    """Executa o download via linha de comando."""
    try:
        download()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
