"""
Yahoo Finance loader (no pdr_override).
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

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=0)
        except Exception:
            pass

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df