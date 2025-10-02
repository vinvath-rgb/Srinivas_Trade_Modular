# srini_mod_backtester/run.py
from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import Dict

from .data_loader import load_adj_close
from .backtest_core import run_backtest

def main():
    st.set_page_config(page_title="US Backtester (Srini) – Minimal", layout="wide")
    st.title("US Backtester (Minimal Working Wire-up)")

    with st.sidebar:
        st.header("Settings")
        tickers = st.text_input("Tickers (comma-separated)", value="SPY,XLK,ACN").upper()
        start = st.text_input("Start date (YYYY-MM-DD)", value="2015-01-01")
        end = st.text_input("End date (YYYY-MM-DD)", value="2025-09-30")

        strat = st.selectbox("Strategy", ["SMA Crossover", "RSI Mean Reversion"])
        long_only = st.checkbox("Long-only", value=True)
        vol_target = st.slider("Vol target (annualized)", 0.05, 0.40, 0.15, 0.01)

        if strat == "SMA Crossover":
            fast = st.number_input("Fast SMA", min_value=5, max_value=200, value=20, step=1)
            slow = st.number_input("Slow SMA", min_value=10, max_value=400, value=100, step=1)
            params = {"fast": fast, "slow": slow, "vol_span": 20}
        else:
            lookback = st.number_input("RSI lookback", min_value=5, max_value=60, value=14, step=1)
            buy_lt = st.number_input("RSI Buy <", min_value=5.0, max_value=50.0, value=30.0, step=1.0)
            sell_gt = st.number_input("RSI Sell >", min_value=50.0, max_value=95.0, value=70.0, step=1.0)
            params = {"lookback": lookback, "buy_lt": buy_lt, "sell_gt": sell_gt, "vol_span": 20}

        run_btn = st.button("Run Backtest")

    if not run_btn:
        st.info("Set your inputs in the sidebar and click **Run Backtest**.")
        return

    # --- Load data ---
    tick_list = [t.strip() for t in tickers.split(",") if t.strip()]
    prices: Dict[str, pd.DataFrame] = load_adj_close(tick_list, start, end)

    if not prices:
        st.error("No data downloaded. Check tickers or date range.")
        return

    # --- Run backtests ---
    curves = {}
    rows = []
    for t, df in prices.items():
        result_df, metrics = run_backtest(
            df=df,
            strategy=strat,
            params=params,
            vol_target=vol_target,
            long_only=long_only,
        )
        curves[t] = result_df["Equity"]
        rows.append({
            "Ticker": t,
            **metrics
        })

    # --- Show results ---
    st.subheader("Equity curves")
    eq = pd.DataFrame(curves).dropna(how="all")
    st.line_chart(eq)

    st.subheader("Summary metrics")
    summary = pd.DataFrame(rows).set_index("Ticker")
    st.dataframe(summary.style.format({"CAGR": "{:.2%}", "Sharpe": "{:.2f}",
                                       "MaxDD": "{:.2%}", "Exposure": "{:.2f}",
                                       "LastEquity": "{:.2f}"}))