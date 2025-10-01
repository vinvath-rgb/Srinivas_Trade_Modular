import numpy as np
import pandas as pd

def annualize_return(returns: pd.Series) -> float:
    if returns.empty: return 0.0
    return float((1 + returns.mean())**252 - 1)

def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    if returns.empty: return 0.0
    ex = returns - rf/252
    sd = ex.std()
    return float((ex.mean() / (sd + 1e-12)) * np.sqrt(252))
