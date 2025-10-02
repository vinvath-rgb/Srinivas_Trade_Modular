# srini_mod_backtester/backtest_core.py
from __future__ import annotations

from typing import Dict, Tuple, Literal
import numpy as np
import pandas as pd

from .indicators import sma, rsi, atr
from .sizing import target_vol_leverage, position
from .utils import annualize_return, sharpe_ratio, max_drawdown

StrategyName = Literal["SMA Crossover", "RSI Mean Reversion"]

def _signal_sma(df: pd.DataFrame, fast: int = 20, slow: int = 100, long_only: bool = True) -> pd.Series:
    f = sma(df["Adj Close"], fast)
    s = sma(df["Adj Close"], slow)
    sig = np.where(f > s, 1.0, (-1.0 if not long_only else 0.0))
    return pd.Series(sig, index=df.index).shift(1).fillna(0.0)

def _signal_rsi(
    df: pd.DataFrame,
    lookback: int = 14,
    buy_lt: float = 30,
    sell_gt: float = 70,
    long_only: bool = True
) -> pd.Series:
    r = rsi(df["Adj Close"], lookback)
    base = np.where(r < buy_lt, 1.0,
                    np.where(r > sell_gt, (-1.0 if not long_only else 0.0), np.nan))
    out = pd.Series(base, index=df.index).ffill().fillna(0.0)
    return out.shift(1).fillna(0.0)

def _apply_exits(
    df: pd.DataFrame,
    raw_pos: pd.Series,
    atr_mult_stop: float | None,
    atr_mult_tp: float | None,
    atr_lookback: int = 14,
) -> pd.Series:
    if (atr_mult_stop is None) and (atr_mult_tp is None):
        return raw_pos

    tr_atr = atr(df["High"], df["Low"], df["Close"], atr_lookback).reindex(df.index).ffill()
    pos = raw_pos.copy().fillna(0.0)
    exec_pos = pos.copy()

    in_pos = 0.0
    entry = np.nan
    for i in range(len(df)):
        px = df["Close"].iat[i]
        desired = pos.iat[i]
        this_atr = tr_atr.iat[i]

        if desired != in_pos:
            in_pos = desired
            entry = px if in_pos != 0 else np.nan
            exec_pos.iat[i] = in_pos
            continue

        if in_pos != 0 and not np.isnan(entry):
            stop_hit = False
            tp_hit = False
            if atr_mult_stop is not None:
                if in_pos > 0 and px <= entry - atr_mult_stop * this_atr:
                    stop_hit = True
                if in_pos < 0 and px >= entry + atr_mult_stop * this_atr:
                    stop_hit = True
            if atr_mult_tp is not None:
                if in_pos > 0 and px >= entry + atr_mult_tp * this_atr:
                    tp_hit = True
                if in_pos < 0 and px <= entry - atr_mult_tp * this_atr:
                    tp_hit = True
            if stop_hit or tp_hit:
                in_pos = 0.0
                entry = np.nan

        exec_pos.iat[i] = in_pos

    return exec_pos

def backtest_one(
    df: pd.DataFrame,
    strategy: StrategyName,
    params: Dict | None = None,
    vol_target: float = 0.15,
    atr_stop: float | None = None,
    take_profit: float | None = None,
    long_only: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    params = params or {}

    df = df.sort_index().copy()
    ret = df["Adj Close"].pct_change().fillna(0.0)

    if strategy == "SMA Crossover":
        fast = int(params.get("fast", 20))
        slow = int(params.get("slow", 100))
        sig = _signal_sma(df, fast=fast, slow=slow, long_only=long_only)
        sig_name = f"SMA[{fast}>{slow}]"
    elif strategy == "RSI Mean Reversion":
        lb = int(params.get("lookback", 14))
        buy_lt = float(params.get("buy_lt", 30))
        sell_gt = float(params.get("sell_gt", 70))
        sig = _signal_rsi(df, lookback=lb, buy_lt=buy_lt, sell_gt=sell_gt, long_only=long_only)
        sig_name = f"RSI[{lb}] {buy_lt}/{sell_gt}"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    span = int(params.get("vol_span", 20))
    lev = target_vol_leverage(ret, vol_target=vol_target, span=span)
    pre_exec_pos = position(sig, lev)

    exec_pos = _apply_exits(df, pre_exec_pos, atr_stop, take_profit)

    pnl = exec_pos * ret
    equity = (1.0 + pnl).cumprod()
    realized_vol = ret.ewm(span=span, adjust=False).std() * np.sqrt(252)
    exposure = float((exec_pos != 0).sum() / len(exec_pos)) if len(exec_pos) else 0.0

    cagr = annualize_return(pnl)
    shrp = sharpe_ratio(pnl)
    maxdd = max_drawdown(equity)

    stats = {
        "Strategy": strategy,
        "Signal": sig_name,
        "VolTarget": float(vol_target),
        "ATR_Stop": (None if atr_stop is None else float(atr_stop)),
        "TP_ATR": (None if take_profit is None else float(take_profit)),
        "Exposure": round(exposure, 3),
        "CAGR": round(float(cagr), 4),
        "Sharpe": round(float(shrp), 2),
        "MaxDD": round(float(maxdd), 4),
        "LastEquity": round(float(equity.iloc[-1]), 4),
        "Bars": int(len(df)),
        "TradesApprox": int((np.abs(np.diff(exec_pos.values)) > 0).sum()),
    }

    out = pd.DataFrame({
        "Return": ret,
        "RealizedVol": realized_vol,
        "Leverage": lev,
        "Signal": sig,
        "PreExecPosition": pre_exec_pos,
        "Position": exec_pos,
        "PnL": pnl,
        "Equity": equity,
    }, index=df.index)

    return out, stats