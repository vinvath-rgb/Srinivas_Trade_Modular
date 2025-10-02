import pandas as pd
import numpy as np

__all__ = ["add_sma", "add_rsi", "add_bbands", "sma", "rsi"]

def add_sma(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    df = df.copy()
    df[f"SMA{window}"] = df.groupby("Ticker")["Close"].transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    return df

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    def _rsi(close: pd.Series, n: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        out = 100 - (100 / (1 + rs))
        return out.fillna(50.0)
    df[f"RSI{window}"] = df.groupby("Ticker")["Close"].transform(lambda s: _rsi(s, window))
    return df

def add_bbands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("Ticker")["Close"]
    ma = g.transform(lambda s: s.rolling(window, min_periods=window).mean())
    sd = g.transform(lambda s: s.rolling(window, min_periods=window).std())
    df["BB_Mid"] = ma
    df["BB_Up"] = ma + num_std * sd
    df["BB_Lo"] = ma - num_std * sd
    return df

# Series helpers (used by signals.py)
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()

def rsi(series: pd.Series, lb: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/lb, min_periods=lb, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/lb, min_periods=lb, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)