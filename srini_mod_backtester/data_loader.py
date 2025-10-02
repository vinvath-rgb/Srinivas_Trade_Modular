# srini_mod_backtester/data_loader.py
from __future__ import annotations

import time
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf


# ------------------------------ helpers --------------------------------------

def _inclusive_end(end_str: str) -> str:
    """yfinance 'end' is exclusive; bump by +1 day."""
    try:
        d = datetime.strptime(end_str, "%Y-%m-%d")
    except ValueError:
        d = pd.to_datetime(end_str).to_pydatetime()
    return (d + timedelta(days=1)).strftime("%Y-%m-%d")


def _yf_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; SriniBacktester/1.0)"})
    return s


def _normalize_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Ensure single-index OHLCV; create Adj Close if missing."""
    if isinstance(df.columns, pd.MultiIndex):
        # (field, ticker) or (ticker, field) — collapse to fields
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        elif ticker in df.columns.get_level_values(0):
            df = df.xs(ticker, axis=1, level=0)

    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    return df[cols].copy()


# ------------------------------ Yahoo ----------------------------------------

def _fetch_yahoo(ticker: str, start: str, end: str, tries: int = 3) -> Tuple[pd.DataFrame, str]:
    sess = _yf_session()
    end_inc = _inclusive_end(end)
    last_err = ""
    for i in range(1, tries + 1):
        try:
            df = yf.download(
                tickers=ticker,
                start=start,
                end=end_inc,
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=False,
                session=sess,
                timeout=30,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                return _normalize_ohlcv(df, ticker), "Yahoo"
            last_err = "Yahoo returned empty dataframe"
        except Exception as e:
            last_err = f"Yahoo error: {type(e).__name__}: {e}"
        time.sleep(0.8 * i)
    return pd.DataFrame(), last_err


# ------------------------------ Stooq via pandas_datareader -------------------

def _fetch_stooq_pdr(ticker: str, start: str, end: str) -> Tuple[pd.DataFrame, str]:
    """Stooq using pandas_datareader (no yfinance override)."""
    try:
        import pandas_datareader.data as pdr
    except Exception as e:
        return pd.DataFrame(), f"Stooq PDR import error: {type(e).__name__}: {e}"

    try:
        df = pdr.get_data_stooq(ticker)
        if df is None or df.empty:
            return pd.DataFrame(), "Stooq PDR returned empty dataframe"

        # Stooq returns descending dates
        df = df.sort_index()

        # Slice by date (inclusive)
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        df = df.loc[(df.index >= s) & (df.index <= e)]

        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]

        keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
        return df[keep].copy(), "Stooq(PDR)"
    except Exception as e:
        return pd.DataFrame(), f"Stooq PDR error: {type(e).__name__}: {e}"


# ------------------------------ Stooq raw CSV (final fallback) ---------------

def _stooq_symbol_csv(ticker: str) -> str:
    """
    CSV endpoint expects symbols like 'spy.us' for US listings.
    Heuristic: if there's already a dot (e.g., BRK.B), leave as-is; else add '.us'.
    """
    t = ticker.strip().lower()
    if "." in t:
        return t
    return f"{t}.us"

def _fetch_stooq_csv(ticker: str, start: str, end: str) -> Tuple[pd.DataFrame, str]:
    sym = _stooq_symbol_csv(ticker)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        df = pd.read_csv(url)
        if df is None or df.empty:
            return pd.DataFrame(), "Stooq CSV returned empty dataframe"

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        # Align columns and add Adj Close if missing
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]

        # Date slice (inclusive)
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        df = df.loc[(df.index >= s) & (df.index <= e)]

        keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
        return df[keep].copy(), "Stooq(CSV)"
    except Exception as e:
        return pd.DataFrame(), f"Stooq CSV error: {type(e).__name__}: {e}"


# ------------------------------ public API -----------------------------------

def load_adj_close(
    tickers: List[str], start: str, end: str
) -> Tuple[Dict[str, pd.DataFrame], List[str], Dict[str, str]]:
    """
    Returns (data, failed, notes):
      data  : {ticker -> OHLCV DataFrame}
      failed: [tickers with no data]
      notes : {ticker -> 'Yahoo' | 'Stooq(PDR)' | 'Stooq(CSV)' | error string}
    """
    out: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []
    notes: Dict[str, str] = {}

    for raw in tickers:
        t = raw.strip().upper()
        if not t:
            continue

        # 1) Try Yahoo
        df, note = _fetch_yahoo(t, start, end)
        if not df.empty:
            out[t] = df
            notes[t] = note
            continue

        # 2) Try Stooq via pandas_datareader
        df, note = _fetch_stooq_pdr(t, start, end)
        if not df.empty:
            out[t] = df
            notes[t] = note
            continue

        # 3) Try raw CSV endpoint
        df, note = _fetch_stooq_csv(t, start, end)
        if not df.empty:
            out[t] = df
            notes[t] = note
            continue

        # 4) All failed
        failed.append(t)
        notes[t] = note

    return out, failed, notes