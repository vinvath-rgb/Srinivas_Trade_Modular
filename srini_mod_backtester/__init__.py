"""
Lightweight package init to avoid importing heavy/legacy modules at import time.
We purposely do NOT import `signals` here.
"""

from .data_loader import get_prices
from .indicators import add_sma, add_rsi, add_bbands
from .backtest_core import equity_curve_long_only

__all__ = [
    "get_prices",
    "add_sma",
    "add_rsi",
    "add_bbands",
    "equity_curve_long_only",
]