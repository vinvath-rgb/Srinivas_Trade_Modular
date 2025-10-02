from __future__ import annotations
import pandas as pd
import numpy as np
from .indicators import sma, rsi

__all__ = ["sma_crossover", "rsi_meanrev"]

def sma_crossover(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    f = sma(close, fast)
    s = sma(close, slow)
    sig = pd.Series(0, index=close.index, dtype=int)
    sig[f > s] = 1
    sig[f < s] = -1
    return sig

def rsi_meanrev(close: pd.Series, lb: int = 14, buy_th: int = 30, sell_th: int = 70) -> pd.Series:
    rv = rsi(close, lb)
    sig = pd.Series(0, index=close.index, dtype=int)
    # enter long when oversold, exit when overbought (shorts not used by UI)
    holding = 0
    for i, dt in enumerate(close.index):
        if rv.iloc[i] >= sell_th:
            holding = 0
        elif rv.iloc[i] <= buy_th:
            holding = 1
        sig.iloc[i] = holding  # 1/0
    return sig