# srini_mod_backtester/data_loader.py
from __future__ import annotations
import pandas as pd
import yfinance as yf
from typing import List, Dict

def load_adj_close(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """
    Returns dict[ticker] -> DataFrame with columns: ['Adj Close']
    Index is DatetimeIndex (business days).
    """
    out: Dict[str, pd.DataFrame] = {}
    if not tickers:
        return out

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=False,
        timeout=60,
    )

    # yfinance multi-index if multiple tickers; single index if one ticker
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data.columns.get_level_values(0):
                df = data[t][["Adj Close"]].dropna().copy()
                df.columns = ["Adj Close"]
                out[t] = df
    else:
        # single ticker case
        df = data[["Adj Close"]].dropna().copy()
        df.columns = ["Adj Close"]
        out[tickers[0]] = df

    return out