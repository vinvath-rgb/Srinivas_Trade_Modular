# srini_mod_backtester/data_loader.py
from __future__ import annotations
import datetime as dt
from typing import Dict, List
import pandas as pd
import yfinance as yf

# --- helpers ---------------------------------------------------------------

def _normalize_prices(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Accepts either a single-level OHLCV frame or a yfinance multi-index frame.
    Returns: columns = [Open, High, Low, Close, Adj Close, Volume] for one ticker.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance multi-index: (field, ticker) or (ticker, field)
        if ticker in df.columns.levels[0]:
            # (ticker, field)
            out = df[ticker].copy()
        elif ticker in df.columns.levels[1]:
            # (field, ticker)
            out = df.xs(ticker, axis=1, level=1).copy()
        else:
            raise KeyError(f"Ticker {ticker} not in downloaded frame.")
    else:
        out = df.copy()

    # Standardize column casing
    rename_map = {c: c.title() for c in out.columns}
    out = out.rename(columns=rename_map)
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col not in out.columns:
            # Some feeds miss 'Adj Close'—fallback to Close
            if col == "Adj Close" and "Close" in out.columns:
                out[col] = out["Close"]
            else:
                out[col] = pd.NA
    return out[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]

# --- public API ------------------------------------------------------------

def load_prices(
    tickers: List[str],
    start: str | dt.date,
    end: str | dt.date,
    source: str = "yahoo",
    threads: bool = False,
    timeout: int = 60,
) -> Dict[str, pd.DataFrame]:
    """
    Download daily OHLCV for each ticker. Currently uses yfinance for all.
    Returns dict: {ticker: DataFrame[Date-indexed OHLCV]}.
    """
    if isinstance(start, str): start = dt.date.fromisoformat(start.replace("/", "-"))
    if isinstance(end, str):   end = dt.date.fromisoformat(end.replace("/", "-"))

    end_inclusive = end + dt.timedelta(days=1)  # yfinance end is exclusive
    got: Dict[str, pd.DataFrame] = {}

    # Use a single batched download for efficiency where possible.
    dl = yf.download(
        tickers=tickers,
        start=start,
        end=end_inclusive,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=threads,
        timeout=timeout,
    )

    for t in tickers:
        try:
            one = _normalize_prices(dl, t)
            one.index = pd.to_datetime(one.index)
            one = one.sort_index()
            # Drop rows with all-NA Close (avoid empty frames)
            one = one[one["Close"].notna()]
            got[t] = one
        except Exception:
            # Fallback: single-ticker call (sometimes more reliable)
            try:
                single = yf.download(
                    t, start=start, end=end_inclusive, interval="1d",
                    auto_adjust=False, progress=False, threads=False, timeout=timeout
                )
                single = _normalize_prices(single, t)
                single.index = pd.to_datetime(single.index)
                single = single.sort_index()
                single = single[single["Close"].notna()]
                if not single.empty:
                    got[t] = single
            except Exception:
                pass

    return got
