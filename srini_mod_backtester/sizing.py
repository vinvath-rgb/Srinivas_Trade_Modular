import numpy as np
import pandas as pd

def target_vol_leverage(returns: pd.Series, vol_target: float, span: int = 20) -> pd.Series:
    vol = returns.ewm(span=span, adjust=False).std() * (252**0.5)
    lev = (vol_target / (vol + 1e-12)).clip(upper=5.0)
    return lev.fillna(0.0)

def position(signal: pd.Series, leverage: pd.Series) -> pd.Series:
    return (signal * leverage).shift(1).fillna(0.0)
