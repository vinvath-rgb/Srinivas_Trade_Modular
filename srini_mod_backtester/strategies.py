import pandas as pd
import numpy as np

__all__ = ["sma_crossover_signals", "rsi_mean_reversion"]


def sma_crossover_signals(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("Ticker")["Close"]
    df[f"SMA{fast}"] = g.transform(lambda s: s.rolling(fast).mean())
    df[f"SMA{slow}"] = g.transform(lambda s: s.rolling(slow).mean())
    df["Signal"] = np.where(df[f"SMA{fast}"] > df[f"SMA{slow}"]], 1, 0)
    df["Trade"] = df.groupby("Ticker")["Signal"].diff().fillna(0)  # 1=buy, -1=sell
    return df


def rsi_mean_reversion(df: pd.DataFrame, low: int = 30, high: int = 70) -> pd.DataFrame:
    df = df.copy()
    if "RSI14" not in df.columns:
        raise ValueError("RSI14 missing, run add_rsi first.")
    df["Signal"] = 0
    df.loc[df["RSI14"] < low, "Signal"] = 1
    df.loc[df["RSI14"] > high, "Signal"] = 0
    df["Signal"] = df.groupby("Ticker")["Signal"].ffill().fillna(0)
    df["Trade"] = df.groupby("Ticker")["Signal"].diff().fillna(0)
    return df
    