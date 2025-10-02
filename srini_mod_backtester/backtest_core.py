# srini_mod_backtester/backtest_core.py
from __future__ import annotations
from typing import Dict, Literal, Tuple
import numpy as np
import pandas as pd

from .indicators import sma, rsi, atr
from .sizing import target_vol_leverage
from .utils import annualize_return, sharpe_ratio

StrategyName = Literal["SMA Crossover", "RSI Mean Reversion"]

def _signals_sma(df: pd.DataFrame, fast:int=20, slow:int=100) -> pd.Series:
    f = sma(df["Adj Close"], fast)
    s = sma(df["Adj Close"], slow)
    sig = np.where(f > s, 1.0, 0.0)  # long-only version
    return pd.Series(sig, index=df.index, name="signal")

def _signals_rsi(df: pd.DataFrame, lookback:int=14, buy:int=30, sell:int=70) -> pd.Series:
    r = rsi(df["Adj Close"], lookback)
    sig = np.where(r < buy, 1.0, np.where(r > sell, 0.0, np.nan))
    # carry forward last decision (hold until opposite)
    sig = pd.Series(sig, index=df.index).ffill().fillna(0.0)
    sig.name = "signal"
    return sig

def backtest_one(
    df: pd.DataFrame,
    strategy: StrategyName = "SMA Crossover",
    params: dict | None = None,
    vol_target: float = 0.15,        # annualized
    atr_stop_x: float | None = None, # e.g. 3.0 => stop = entry - 3*ATR
) -> Tuple[pd.DataFrame, dict]:
    """
    Runs a single-ticker backtest.
    Returns (daily_frame, summary_stats)
    daily_frame columns: ['ret','signal','lev','pnl','equity','exposure', ...]
    """
    params = params or {}
    px = df["Adj Close"].astype(float)
    ret = px.pct_change().fillna(0.0)

    # --- signals
    if strategy == "SMA Crossover":
        fast = int(params.get("fast", 20))
        slow = int(params.get("slow", 100))
        sig = _signals_sma(df, fast=fast, slow=slow)
    elif strategy == "RSI Mean Reversion":
        look = int(params.get("lookback",14))
        buy = int(params.get("buy",30))
        sell = int(params.get("sell",70))
        sig = _signals_rsi(df, lookback=look, buy=buy, sell=sell)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # --- sizing: simple daily realized vol -> leverage
    # realized vol (rolling) on returns
    realized_vol = ret.rolling(20).std() * np.sqrt(252)
    lev = target_vol_leverage(realized_vol, vol_target)  # user function in sizing.py
    lev = lev.fillna(0.0)

    # position before stop logic
    pos_raw = sig * lev

    # --- optional ATR stop (long-only; simple example)
    if atr_stop_x:
        true_range = atr(df["High"], df["Low"], df["Close"], window=14)
        stop = pd.Series(np.nan, index=px.index)
        in_trade = False
        entry = np.nan
        for i in range(1, len(px)):
            if not in_trade and pos_raw.iloc[i-1] > 0 and pos_raw.iloc[i] > 0:
                # enter
                in_trade = True
                entry = px.iloc[i]
                stop.iloc[i] = entry - atr_stop_x * true_range.iloc[i]
            elif in_trade:
                # trail stop up if price rises
                new_stop = px.iloc[i] - atr_stop_x * true_range.iloc[i]
                stop.iloc[i] = max(stop.iloc[i-1], new_stop) if not np.isnan(stop.iloc[i-1]) else new_stop
                # exit if broken
                if px.iloc[i] <= stop.iloc[i]:
                    in_trade = False
                    entry = np.nan
            else:
                stop.iloc[i] = stop.iloc[i-1] if i>0 else np.nan
        # zero out position when stop is active and price below stop
        stopped = (px <= stop) & stop.notna()
        pos = pos_raw.where(~stopped, 0.0)
    else:
        pos = pos_raw

    # --- PnL (simple)
    pnl = (pos.shift(1).fillna(0.0)) * ret  # yesterday’s position * today’s return
    equity = (1.0 + pnl).cumprod()
    exposure = (pos.abs() > 1e-9).astype(float)

    # --- stats
    cagr = annualize_return(equity.pct_change().fillna(0.0))
    shrp = sharpe_ratio(pnl)
    maxdd = (equity / equity.cummax() - 1.0).min()
    exp = exposure.mean()

    daily = pd.DataFrame({
        "ret": ret,
        "signal": sig,
        "lev": pos,
        "pnl": pnl,
        "equity": equity,
        "exposure": exposure,
        "realized_vol": realized_vol,
    })

    stats = {
        "CAGR": float(cagr),
        "Sharpe": float(shrp),
        "MaxDD": float(maxdd),
        "Exposure": float(exp),
        "LastEquity": float(equity.iloc[-1]),
    }
    return daily, stats

def backtest_multi(
    price_map: Dict[str, pd.DataFrame],
    strategy: StrategyName = "SMA Crossover",
    params: dict | None = None,
    vol_target: float = 0.15,
    atr_stop_x: float | None = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Runs backtest for multiple tickers.
    Returns: (per_ticker_daily_frames, summary_df)
    """
    per: Dict[str, pd.DataFrame] = {}
    rows = []
    for t, df in price_map.items():
        if df is None or df.empty:
            continue
        daily, s = backtest_one(
            df, strategy=strategy, params=params,
            vol_target=vol_target, atr_stop_x=atr_stop_x
        )
        per[t] = daily
        rows.append({"Ticker": t, **s})
    summary = pd.DataFrame(rows).set_index("Ticker")
    return per, summary
