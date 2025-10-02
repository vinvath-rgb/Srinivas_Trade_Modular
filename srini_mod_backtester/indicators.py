# srini_mod_backtester/indicators.py
from __future__ import annotations
import pandas as pd

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()

def rsi(series: pd.Series, lookback: int = 14) -> pd.Series:
    # Wilder RSI
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/lookback, adjust=False).mean()
    roll_down = down.ewm(alpha=1/lookback, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val