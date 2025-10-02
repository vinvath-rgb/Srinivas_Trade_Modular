# srini_mod_backtester/data_loader.py
from __future__ import annotations

import time
from typing import Dict, List, Tuple
import pandas as pd
import requests
import yfinance as yf

# ---------- helpers -----------------------------------------------------------

def _yf_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; SriniBacktester/1.0)"})
    return s

def _normalize(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return Yahoo-like OHLCV single-index columns."""
    if isinstance(df.columns, pd.MultiIndex):
        # either (field, ticker) or (ticker, field)
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        elif ticker in df.columns.get_level_values(0):
            df = df.xs(ticker, axis=1, level=0)

    # make sure expected cols exist
    want = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    cols = [c for c in want if c in df.columns]
    return df[cols].copy()

# ---------- sources -----------------------------------------------------------

def _fetch_yahoo(ticker: str, start: str, end: str, tries: int = 3) -> Tuple[pd.DataFrame, str]:
    sess = _yf_session()
    last_err = ""
    for i in range(tries):
        try:
            df = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=False,
                timeout=30,
                session=sess,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                return _normalize(df, ticker), ""
            last_err = "empty dataframe"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(1.0 + i)
    return pd.DataFrame(), f"Yahoo fail {ticker}: {last_err}"

def _fetch_stooq(ticker: str, start: str, end: str) -> Tuple[pd.DataFrame, str]:
    try:
        import pandas_datareader.data as pdr
        yf.pdr_override()
        df = pdr.get_data_stooq(ticker)
        if df is None or df.empty:
            return pd.DataFrame(), "Stooq empty"
        df = df.sort_index()
        df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
        return df[keep].copy(), ""
    except Exception as e:
        return pd.DataFrame(), f"Stooq error: {type(e).__name__}: {e}"

# ---------- public API --------------------------------------------------------

def load_adj_close(tickers: List[str], start: str, end: str) -> Tuple[Dict[str, pd.DataFrame], List[str], Dict[str, str]]:
    """
    Returns (data, failed, notes)
      data  : {ticker -> OHLCV DataFrame}
      failed: [tickers with no data]
      notes : {ticker -> 'Yahoo' or 'Stooq' source or error message}
    """
    out: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []
    notes: Dict[str, str] = {}

    for raw in tickers:
        t = raw.strip().upper()
        if not t:
            continue

        df, err = _fetch_yahoo(t, start, end)
        if not df.empty:
            out[t] = df
            notes[t] = "Yahoo"
            continue

        df, err2 = _fetch_stooq(t, start, end)
        if not df.empty:
            out[t] = df
            notes[t] = "Stooq"
            continue

        failed.append(t)
        notes[t] = err2 or err or "unknown"

    return out, failed, notes