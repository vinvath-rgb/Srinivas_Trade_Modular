"""
data_loader.py
---------------
Price loader with Yahoo primary, Stooq fallback.
"""

from functools import lru_cache
from typing import Optional
import pandas as pd
import yfinance as yf

try:
    import pandas_datareader.data as web
    HAS_PDR = True
except ImportError:
    HAS_PDR = False


@lru_cache(maxsize=256)
def load_prices(
    ticker: str,
    start: str,
    end: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Try Yahoo first, fallback to Stooq if Yahoo fails/returns empty.
    """
    # --- Try Yahoo ---
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
    )

    if isinstance(df, pd.DataFrame) and not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(ticker, axis=1, level=0)
            except Exception:
                pass
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        return df

    # --- Fallback: Stooq ---
    if HAS_PDR:
        try:
            df = web.DataReader(ticker, "stooq", start=start, end=end)
            df = df.sort_index()
            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"
            return df
        except Exception as e:
            print(f"Stooq fallback failed for {ticker}: {e}")

    # --- If all failed ---
    return pd.DataFrame()