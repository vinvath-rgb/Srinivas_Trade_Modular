import pandas as pd

def apply_returns(prices: pd.Series, position: pd.Series) -> pd.Series:
    rets = prices.pct_change().fillna(0.0)
    return position * rets
