import pandas as pd
import numpy as np

__all__ = ["sma_crossover_signals", "rsi_mean_reversion"]

def sma_crossover_strategy(df, fast=10, slow=50):
    df[f"SMA{fast}"] = df["Close"].rolling(window=fast).mean()
    df[f"SMA{slow}"] = df["Close"].rolling(window=slow).mean()
    df["Signal"] = np.where(df[f"SMA{fast}"] > df[f"SMA{slow}"], 1, 0)
    df["Position"] = df["Signal"].diff()
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
    