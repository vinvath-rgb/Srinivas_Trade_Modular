import pandas as pd
from .indicators import sma, rsi

def sma_crossover(prices: pd.Series, fast: int = 20, slow: int = 100) -> pd.Series:
    f = sma(prices, fast); s = sma(prices, slow)
    sig = pd.Series(0.0, index=prices.index)
    sig[f > s] = 1.0; sig[f < s] = -1.0
    return sig.fillna(0.0)

def rsi_meanrev(prices: pd.Series, lb=14, buy_th=30, sell_th=70) -> pd.Series:
    rr = rsi(prices, lb=lb); sig = pd.Series(0.0, index=prices.index)
    sig[rr < buy_th] = 1.0; sig[rr > sell_th] = -1.0
    return sig.fillna(0.0)
