#srini_mod_backtester/data_loader.py
from __future__ import annotations

import os
import time
import datetime as dt
from typing import Dict, List

import requests
import pandas as pd
import yfinance as yf


# ---- Robust Yahoo → Stooq fetch ---------------------------------------------

def _yf_session():
    s = yf.utils.get_yf_ratelimit_session()  # has backoff
    # add UA to reduce chance of HTML error pages
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; SriniBacktester/1.0)"})
    return s

def _fetch_yahoo(ticker: str, start: str, end: str, tries: int = 3) -> pd.DataFrame:
    """
    Try Yahoo a few times with a custom session.
    Returns a single-ticker OHLCV frame (columns: Open, High, Low, Close, Adj Close, Volume).
    """
    sess = _yf_session()
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
                threads=False,     # safer on some hosts
                timeout=30,
                session=sess,
            )
            if isinstance(df.columns, pd.MultiIndex):
                # sometimes yfinance returns (field, ticker)
                if ticker in df.columns.levels[-1]:
                    df = df.xs(ticker, axis=1, level=-1)
            if not df.empty and "Adj Close" in df.columns:
                return df
        except Exception:
            # swallow and retry
            pass
        time.sleep(1.0 + i)
    return pd.DataFrame()

def _fetch_stooq(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fallback via Stooq using yfinance's built-in pdr override (stable for many US ETFs).
    """
    try:
        import pandas_datareader.data as pdr
        yf.pdr_override()
        df = pdr.get_data_stooq(ticker)  # Stooq ignores date filter; we’ll slice below
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.sort_index()  # stooq returns desc
        # Align Yahoo-like columns
        if "Close" in df.columns and "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]
        # slice dates
        df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
        return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    except Exception:
        return pd.DataFrame()

# ---- Public loader -----------------------------------------------------------

def load_adj_close(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """
    Returns: dict {ticker: OHLCV DataFrame}
    Tries Yahoo first with retries, then Stooq.
    """
    out: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []

    for t in tickers:
        t = t.strip()
        if not t:
            continue

        df = _fetch_yahoo(t, start, end)
        if df.empty:
            df = _fetch_stooq(t, start, end)

        if df.empty:
            failed.append(t)
        else:
            # Keep only the standard set (and ensure dtype/cols)
            cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
            out[t] = df[cols].copy()

    if failed:
        # Let caller show a friendly message; don’t crash the app
        print(f"Yahoo/Stooq returned no data for: {failed}")

    return out