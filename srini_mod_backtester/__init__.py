"""
Package init kept light. Aliases loader for backwards compatibility.
"""

from .data_loader import load_prices_yahoo as get_prices
from .indicators import add_sma, add_rsi, add_bbands
from .backtest_core import equity_curve_long_only

__all__ = [
    "get_prices",
    "add_sma",
    "add_rsi",
    "add_bbands",
    "equity_curve_long_only",
]