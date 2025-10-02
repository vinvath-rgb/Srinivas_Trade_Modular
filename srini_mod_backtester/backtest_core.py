# srini_mod_backtester/data_loader.py
from __future__ import annotations

import time
from typing import Dict, List, Optional
from datetime import datetime
import requests
import pandas as pd
import yfinance as yf


# ---------- helpers -----------------------------------------------------------

def _yf_session() -> requests.Session:
    """
    Build a plain requests.Session with a desktop User-Agent.
    Newer yfinance versions no longer expose utils.get_yf_ratelimit_session().
    """
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s


def _normalize_prices(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Accepts either a single-level OHLCV frame or a yfinance multi-index frame.
    Returns: columns = [Open, High, Low, Close, Adj Close, Volume] for one ticker.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance multi-index: (field, ticker) or (ticker, field)
        if ticker in df.columns.levels[1]:
            out = df.xs(ticker, axis=1, level=1).copy()       # (field, ticker)
        elif ticker in df.columns.levels[0]:
            out = df[ticker].copy()                            # (ticker, field)
        else:
            raise KeyError(f"Ticker {ticker} not in downloaded frame.")
    else:
        out = df.copy()

    # Standardize column names/casing
    rename_map = {c: c.title() for c in out.columns}
    out = out.rename(columns=rename_map)

    # Ensure all required columns exist
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col not in out.columns:
            if col == "Adj Close" and "Close" in out.columns:
                out[col] = out["Close"]
            else:
                out[col] = pd.NA

    # Order columns
    out = out[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    return out


def _to_datestr(x: str | datetime) -> str:
    if isinstance(x, datetime):
        return x.strftime("%Y-%m-%d")
    return str(x)


# ---------- Yahoo (primary) ---------------------------------------------------

def _fetch_yahoo(
    ticker: str,
    start: str | datetime,
    end: str | datetime,
    tries: int = 2,
    pause: float = 1.0,
) -> pd.DataFrame:
    """
    Robust single-ticker fetch from Yahoo using yfinance with a custom Session.
    Returns a normalized OHLCV DataFrame (may be empty if nothing returned).
    """
    s = _yf_session()
    start_s = _to_datestr(start)
    end_s = _to_datestr(end)

    last_exc: Optional[Exception] = None
    for _ in range(max(1, tries)):
        try:
            df = yf.download(
                tickers=ticker,
                start=start_s,
                end=end_s,
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=False,           # safer on some hosts
                session=s,
            )
            if df is None or len(df) == 0:
                # sometimes Yahoo responds but with no rows
                last_exc = None
            else:
                df = _normalize_prices(df, ticker)
                # Clean index
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                return df
        except Exception as e:
            last_exc = e
        time.sleep(pause)

    # If we get here, either empty or error every time
    if last_exc:
        print(f"[Yahoo] {ticker} failed: {last_exc}")
    return pd.DataFrame()


# ---------- Stooq (fallback) --------------------------------------------------

def _stooq_symbol(ticker: str) -> str:
    """
    Build a Stooq symbol. For most US tickers: lower-case + '.us' (e.g., spy.us).
    This is a heuristic; Stooq coverage is limited for non-US markets.
    """
    t = ticker.strip().lower()
    if "." in t:   # e.g., 'brk.b' or exchange suffixes — leave as is and try
        return t
    return f"{t}.us"


def _fetch_stooq(
    ticker: str,
    start: str | datetime,
    end: str | datetime,
) -> pd.DataFrame:
    """
    Simple Stooq fetch via CSV. Returns normalized OHLCV (Adj Close = Close).
    """
    sym = _stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        df = pd.read_csv(url)
        if df.empty:
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        # rename to standard
        df = df.rename(columns={
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        })
        # fill Adj Close = Close
        df["Adj Close"] = df["Close"]
        df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]

        # date slice to requested range
        df = df.loc[_to_datestr(start):_to_datestr(end)]
        return df
    except Exception as e:
        print(f"[Stooq] {ticker} failed: {e}")
        return pd.DataFrame()


# ---------- Public API --------------------------------------------------------

def load_adj_close(
    tickers: List[str],
    start: str | datetime,
    end: str | datetime,
    prefer: str = "yahoo",  # "yahoo" -> Stooq fallback
) -> Dict[str, pd.DataFrame]:
    """
    Fetch daily OHLCV for each ticker between start and end (inclusive range),
    normalized to columns: [Open, High, Low, Close, Adj Close, Volume].

    Returns: dict[ticker] -> DataFrame (may be empty if all sources failed).
    """
    out: Dict[str, pd.DataFrame] = {}

    for t in tickers:
        t = t.strip()
        df = pd.DataFrame()

        if prefer.lower() == "yahoo":
            df = _fetch_yahoo(t, start, end)
            if df.empty:
                # fallback to Stooq (best effort for US symbols)
                df = _fetch_stooq(t, start, end)
        else:
            # Stooq first, then Yahoo
            df = _fetch_stooq(t, start, end)
            if df.empty:
                df = _fetch_yahoo(t, start, end)

        if df.empty:
            print(f"[DataLoader] No data for {t} from Yahoo or Stooq.")
        out[t] = df

    return out