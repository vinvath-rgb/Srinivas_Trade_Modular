"""
data_loader.py
---------------
Single-responsibility data loader for price history.

- Uses yfinance WITHOUT pdr_override (removed in new yfinance).
- Returns a clean OHLCV DataFrame indexed by Date.
- Caches results to avoid re-downloading during UI tweaks.
"""

from functools import lru_cache
from typing import Optional
import pandas as pd
import yfinance as yf


@lru_cache(maxsize=256)
def load_prices_yahoo(
    ticker: str,
    start: str,
    end: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Symbol like "SPY".
    start : str (YYYY-MM-DD)
    end   : str (YYYY-MM-DD) or None
    interval : one of {"1d","1wk","1mo"}
    auto_adjust : bool
        Adjust OHLC for splits/dividends.

    Returns
    -------
    pd.DataFrame
        Columns: [Open, High, Low, Close, Adj Close, Volume]
        Index name: "Date"
        Empty DataFrame if nothing returned.
    """
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
    )

    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    # yfinance sometimes returns a column MultiIndex for multi-ticker queries.
    # We force a flat frame for single ticker usage.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=0)

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df