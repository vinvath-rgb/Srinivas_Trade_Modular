# srini_mod_backtester/indicators.py
from __future__ import annotations
import pandas as pd

__all__ = ["atr", "sma", "rsi"]

def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    fillna: bool = False,
) -> pd.Series:
    """
    Wilder's Average True Range.
    Returns a Series named 'ATR_{period}'.
    """
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    out = true_range.ewm(alpha=1 / period, adjust=False).mean()
    if fillna:
        out = out.fillna(0.0)
    return out.rename(f"ATR_{period}")

def sma(
    series: pd.Series,
    period: int = 20,
    min_periods: int | None = None,
    fillna: bool = False,
) -> pd.Series:
    """
    Simple Moving Average.
    Returns a Series named 'SMA_{period}'.
    """
    s = pd.to_numeric(series, errors="coerce")
    if min_periods is None:
        min_periods = period
    out = s.rolling(window=period, min_periods=min_periods).mean()
    if fillna:
        out = out.fillna(method="backfill")
    return out.rename(f"SMA_{period}")

def rsi(
    close: pd.Series,
    period: int = 14,
    method: str = "wilder",  # "wilder" (EMA with alpha=1/period) or "sma"
    fillna: bool = False,
) -> pd.Series:
    """
    Relative Strength Index (RSI).
    Returns a Series named 'RSI_{period}'.
    """
    c = pd.to_numeric(close, errors="coerce")
    delta = c.diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    if method.lower() == "sma":
        avg_gain = gain.rolling(period, min_periods=period).mean()
        avg_loss = loss.rolling(period, min_periods=period).mean()
    else:
        # Wilder's smoothing
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi_series = 100 - (100 / (1 + rs))

    # When avg_loss is 0, RSI should be 100; when avg_gain is 0, RSI should be 0
    rsi_series = rsi_series.fillna(
        value=100.0
    ).where(avg_gain.notna() | avg_loss.notna(), other=pd.NA)

    if fillna:
        rsi_series = rsi_series.fillna(method="backfill")

    return rsi_series.rename(f"RSI_{period}")