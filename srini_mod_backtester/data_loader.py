# srini_mod_backtester/data_loader.py
from __future__ import annotations

import time
from typing import Dict, List
from datetime import datetime, timedelta

import requests
import pandas as pd
import yfinance as yf


# ------------------------------ helpers --------------------------------------

def _bump_end_inclusive(end_str: str) -> str:
    """
    yfinance treats 'end' as exclusive. Bump by +1 day so the user's chosen end
    day is included (if data exists).
    """
    try:
        d = datetime.strptime(end_str, "%Y-%m-%d")
    except ValueError:
        # fall back to common alt formats (e.g. YYYY/MM/DD)
        d = pd.to_datetime(end_str).to_pydatetime()
    return (d + timedelta(days=1)).strftime("%Y-%m-%d")


def _yf_session() -> requests.Session:
    """
    Simple requests session with a friendly UA. Using this (vs yf.utils.*)
    avoids version drift problems on hosted platforms.
    """
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; SriniBacktester/1.0)"})
    return s


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure standard OHLCV columns exist and order them.
    If Adj Close missing, fall back to Close.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # In rare cases a MultiIndex can leak through for single ticker
        df = df.copy()
        # try to collapse if it's (field, ticker)
        first_lvl = df.columns.get_level_values(0).unique().tolist()
        if set(["Open", "High", "Low", "Close", "Adj Close", "Volume"]).intersection(first_lvl):
            df = df.droplevel(-1, axis=1)

    out = df.copy()
    if "Adj Close" not in out.columns and "Close" in out.columns:
        out["Adj Close"] = out["Close"]

    cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in out.columns]
    return out[cols]


# -------------------------- primary fetchers ----------------------------------

def _fetch_yahoo(ticker: str, start: str, end: str, tries: int = 3) -> pd.DataFrame:
    """
    Try Yahoo a few times with a custom session.
    Returns: single-ticker OHLCV DataFrame (may be empty).
    """
    sess = _yf_session()
    end_inc = _bump_end_inclusive(end)

    for attempt in range(1, tries + 1):
        try:
            df = yf.download(
                tickers=ticker,
                start=start,
                end=end_inc,          # inclusive end
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="column",    # avoid MultiIndex for single ticker
                threads=False,        # safer on some hosts
                session=sess,
            )
            if df is None or df.empty:
                # Retry: sometimes first hit returns HTML or empty frame
                time.sleep(0.8 * attempt)
                continue
            df = _normalize_ohlcv(df)
            if not df.empty:
                return df
        except Exception as e:
            # Log and retry
            print(f"[Yahoo] {ticker} attempt {attempt} failed: {e}")
            time.sleep(0.8 * attempt)

    return pd.DataFrame()


def _fetch_stooq(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fallback via Stooq using pandas-datareader.
    Make sure 'pandas-datareader' is in requirements.txt.
    """
    try:
        import pandas_datareader.data as pdr
        yf.pdr_override()  # required by pdr for yfinance-compatible behavior

        df = pdr.get_data_stooq(ticker)  # Stooq ignores our date filter; slice after
        if df is None or df.empty:
            return pd.DataFrame()

        # Stooq returns descending dates
        df = df.sort_index()

        # Align columns
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]

        # Date slice (inclusive)
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        df = df.loc[(df.index >= s) & (df.index <= e)]

        keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
        return df[keep]
    except Exception as e:
        print(f"[Stooq] {ticker} failed: {e}")
        return pd.DataFrame()


# ------------------------------ public API ------------------------------------

def load_adj_close(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for each ticker.
      - Try Yahoo with retries
      - Fallback to Stooq
    Returns: {ticker: DataFrame}
    Never raises on fetch failure; logs and skips empty symbols.
    """
    out: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []

    for raw in tickers:
        t = raw.strip().upper()
        if not t:
            continue

        df = _fetch_yahoo(t, start, end)
        if df.empty:
            df = _fetch_stooq(t, start, end)

        if df.empty:
            failed.append(t)
        else:
            out[t] = df.copy()

    if failed:
        print(f"[DataLoader] No data from Yahoo/Stooq for: {failed}")

    return out