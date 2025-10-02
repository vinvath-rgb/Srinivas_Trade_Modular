import pandas as pd
import numpy as np

__all__ = ["equity_curve_long_only"]


def equity_curve_long_only(
    df: pd.DataFrame,
    init_cash: float = 100_000.0,
    allow_cash_return: float = 0.0,
) -> pd.DataFrame:
    """
    Vectorized daily rebalanced long-only.
    - Uses previous day's Signal as today's position (avoid lookahead).
    - Equal-weight across active signals; if none -> cash.
    Requires: ['Ticker','Date','Close','Signal']
    Returns: ['Date','DailyReturn','Equity']
    """
    req = {"Ticker", "Date", "Close", "Signal"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Input df missing columns: {missing}")

    d = df.copy().sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Per-ticker simple returns
    d["Ret"] = d.groupby("Ticker")["Close"].pct_change().fillna(0.0)
    # Use yesterday's signal as today's position
    d["Position"] = d.groupby("Ticker")["Signal"].shift(1).fillna(0.0)

    def _daily_ret(g: pd.DataFrame) -> float:
        pos = g["Position"].values
        rets = g["Ret"].values
        n = np.sum(pos)
        if n <= 0:
            return allow_cash_return
        w = pos / n
        return float(np.sum(w * rets))

    daily = d.groupby("Date").apply(_daily_ret).rename("DailyReturn").reset_index()
    daily["Equity"] = (1.0 + daily["DailyReturn"]).cumprod() * init_cash
    return daily