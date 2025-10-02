# srini_mod_backtester/backtest_core.py
from __future__ import annotations
from typing import Dict, Tuple, Literal
import numpy as np
import pandas as pd

from .indicators import sma, rsi
from .sizing import target_vol_leverage, position

StrategyName = Literal["SMA Crossover", "RSI Mean Reversion", "SMA+RSI (Composite)"]

# ---------- signal builders ----------

def _signal_sma(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 100,
    long_only: bool = True
) -> pd.Series:
    px = df["Adj Close"]
    f = sma(px, fast)
    s = sma(px, slow)
    sig = np.where(f > s, 1.0, -1.0)
    if long_only:
        sig = np.where(f > s, 1.0, 0.0)
    return pd.Series(sig, index=px.index, name="signal")


def _signal_rsi(
    df: pd.DataFrame,
    lookback: int = 14,
    buy_lt: float = 30,
    sell_gt: float = 70,
    long_only: bool = True
) -> pd.Series:
    px = df["Adj Close"]
    r = rsi(px, lookback)
    if long_only:
        # enter when oversold; hold until opposite
        raw = np.where(r < buy_lt, 1.0, 0.0)
    else:
        raw = np.where(r < buy_lt, 1.0, np.where(r > sell_gt, -1.0, 0.0))

    sig = pd.Series(raw, index=px.index).replace(0.0, np.nan).ffill().fillna(0.0)
    return sig.rename("signal")


def _signal_sma_rsi_composite(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 100,
    rsi_lookback: int = 14,
    buy_lt: float = 30,
    sell_gt: float = 70,
    long_only: bool = True
) -> pd.Series:
    """
    Long when SMA regime is bullish (fast>slow) **and** RSI < buy_lt.
    Exit when RSI > sell_gt or regime turns off (fast<=slow).
    For now we keep it long-only for clarity. (We can add short rules later.)
    """
    px = df["Adj Close"]
    f = sma(px, fast)
    s = sma(px, slow)
    regime = f > s

    r = rsi(px, rsi_lookback)

    entries = regime & (r < buy_lt)
    exits   = (~regime) | (r > sell_gt)

    # Stateful position via forward-fill of toggles:
    sig = pd.Series(np.nan, index=px.index, dtype=float)
    sig[entries] = 1.0
    sig[exits]   = 0.0
    sig = sig.ffill().fillna(0.0)

    if not long_only:
        # (Optional) Add a mirrored short side if you want:
        # shorts when regime is bearish & RSI > sell_gt, exit when regime flips or RSI < buy_lt
        pass

    return sig.rename("signal")

# ---------- backtest engine ----------

def backtest_one(
    df: pd.DataFrame,
    strategy: StrategyName,
    params: Dict,
    vol_target: float = 0.15,
    long_only: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Returns per-ticker result frame (with equity) and metrics dict.
    """
    px = df["Adj Close"].dropna()
    ret = px.pct_change().fillna(0.0)

    # --- signals ---
    if strategy == "SMA Crossover":
        sig = _signal_sma(
            df,
            fast=int(params.get("fast", 20)),
            slow=int(params.get("slow", 100)),
            long_only=long_only,
        )
    elif strategy == "RSI Mean Reversion":
        sig = _signal_rsi(
            df,
            lookback=int(params.get("lookback", 14)),
            buy_lt=float(params.get("buy_lt", 30)),
            sell_gt=float(params.get("sell_gt", 70)),
            long_only=long_only,
        )
    else:  # "SMA+RSI (Composite)"
        sig = _signal_sma_rsi_composite(
            df,
            fast=int(params.get("fast", 20)),
            slow=int(params.get("slow", 100)),
            rsi_lookback=int(params.get("rsi_lookback", 14)),
            buy_lt=float(params.get("buy_lt", 30)),
            sell_gt=float(params.get("sell_gt", 70)),
            long_only=long_only,
        )

    # --- sizing ---
    lev = target_vol_leverage(ret, vol_target=vol_target, span=int(params.get("vol_span", 20)))
    pos = position(sig, lev)

    # --- P&L / equity ---
    strat_ret = pos * ret
    equity = (1.0 + strat_ret).cumprod()

    # metrics
    def ann(ret_s: pd.Series) -> float:
        return float((1 + ret_s).prod() ** (252/len(ret_s)) - 1) if len(ret_s) > 0 else 0.0

    def sharpe(ret_s: pd.Series) -> float:
        mu = ret_s.mean() * 252
        sd = ret_s.std() * np.sqrt(252) + 1e-12
        return float(mu / sd)

    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    maxdd = float(dd.min())

    exposure = float((pos != 0).sum() / max(1, len(pos)))

    metrics = {
        "CAGR": ann(strat_ret),
        "Sharpe": sharpe(strat_ret),
        "MaxDD": maxdd,
        "Exposure": exposure,
        "LastEquity": float(equity.iloc[-1]) if len(equity) else 1.0,
    }

    out = pd.DataFrame({
        "Price": px,
        "Returns": ret,
        "Signal": sig,
        "Leverage": lev,
        "Position": pos,
        "StratRet": strat_ret,
        "Equity": equity,
    })

    return out, metrics

# Backward-compatible alias
def run_backtest(*args, **kwargs):
    return backtest_one(*args, **kwargs)