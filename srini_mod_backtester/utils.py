# srini_mod_backtester/utils.py
import numpy as np
import pandas as pd

def annualize_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualized return from daily returns.
    """
    compounded = (1 + returns).prod()
    n_periods = returns.shape[0]
    return compounded ** (periods_per_year / n_periods) - 1

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Sharpe ratio: mean excess return / std dev of returns.
    """
    excess = returns - risk_free_rate / periods_per_year
    mean = excess.mean() * periods_per_year
    std = excess.std() * np.sqrt(periods_per_year)
    return mean / std if std > 0 else np.nan
