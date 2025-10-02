# srini_mod_backtester/utils.py
import pandas as pd
import numpy as np

__all__ = ["annualize_return", "sharpe_ratio", "max_drawdown"]

def annualize_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualized return based on periodic returns.
    """
    compounded_growth = (1 + returns).prod()
    n_periods = returns.shape[0]
    return compounded_growth ** (periods_per_year / n_periods) - 1

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Sharpe ratio: excess return / volatility.
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    ann_excess_return = annualize_return(excess_returns, periods_per_year)
    ann_volatility = returns.std() * np.sqrt(periods_per_year)
    if ann_volatility == 0:
        return np.nan
    return ann_excess_return / ann_volatility

def max_drawdown(returns: pd.Series) -> float:
    """
    Max drawdown of an equity curve (cumulative returns).
    Returns a negative float (e.g., -0.25 for -25%).
    """
    # Convert periodic returns to equity curve
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()