"""
US Backtester · Modular (Streamlit UI)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
from datetime import date

# --- use package modules ---
from srini_mod_backtester.data_loader import load_prices
from srini_mod_backtester.signals import sma_crossover, rsi_meanrev
from srini_mod_backtester.backtest_core import equity_curve_long_only  # your existing engine


# ---------- metrics helpers ----------
def _cagr(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    total = float(equity.iloc[-1] / equity.iloc[0])
    yrs = max(len(equity) / 252.0, 1e-9)
    return total ** (1 / yrs) - 1.0

def _sharpe(daily_ret: pd.Series, rf_daily: float = 0.0) -> float:
    excess = daily_ret - rf_daily
    if len(excess) < 2 or excess.std(ddof=0) == 0:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / excess.std(ddof=0))

def _maxdd(equity: pd.Series) -> float:
    roll = equity.cummax()
    dd = equity / roll - 1.0
    return float(dd.min()) if not dd.empty else 0.0


# ---------- page ----------
st.set_page_config(page_title="US Backtester · Modular", layout="wide")
st.title("US Backtester · Modular")

with st.sidebar:
    st.subheader("Data")
    tickers_raw = st.text_input("Tickers (comma-separated)", value="SPY,XLK,ACN")
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start", value=pd.to_datetime("2015-01-01"))
    with c2:
        end_date = st.date_input("End", value=date.today())
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

    st.subheader("Strategy")
    strategy = st.radio("Pick strategy", ["SMA crossover", "RSI mean reversion"], index=0)
    fast = st.number_input("SMA fast", value=20, step=1, min_value=1)
    slow = st.number_input("SMA slow", value=50, step=1, min_value=2)
    rsi_lb = st.number_input("RSI length", value=14, step=1, min_value=2)
    rsi_buy = st.number_input("RSI Buy <=", value=30, step=1, min_value=1, max_value=99)
    rsi_sell = st.number_input("RSI Sell >=", value=70, step=1, min_value=1, max_value=99)

    st.subheader("Backtest")
    initial_cash = st.number_input("Initial Cash", value=100_000, step=1_000, min_value=0)
    cash_daily_return = st.number_input("Cash Daily Return (decimal)", value=0.0, step=0.0001, format="%.4f")

    run_clicked = st.button("Run Backtest", type="primary", use_container_width=True)

status = st.empty()

# ---------- run ----------
if run_clicked:
    status.info("Downloading data & running backtest…")

    rows = []
    for t in tickers:
        px = load_prices_yahoo(t, str(start_date), str(end_date), interval=interval)
        if px.empty or "Close" not in px.columns:
            st.error(f"Data load failed for {t}. Check symbol/date range/interval.")
            st.stop()

        close = px["Close"].astype(float)

        if strategy.lower().startswith("sma"):
            sig_raw = sma_crossover(close, fast=fast, slow=slow)  # -1/0/1
        else:
            sig_raw = rsi_meanrev(close, lb=rsi_lb, buy_th=rsi_buy, sell_th=rsi_sell)  # -1/0/1

        # long-only: map shorts to 0
        sig_long_only = (sig_raw > 0).astype(float)

        df_t = pd.DataFrame({
            "Ticker": t,
            "Date": close.index,
            "Close": close.values,
            "Signal": sig_long_only.values,
        })
        rows.append(df_t)

    panel = pd.concat(rows, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])

    daily = equity_curve_long_only(
        panel,
        init_cash=float(initial_cash),
        allow_cash_return=float(cash_daily_return),
    )
    if "Equity" not in daily.columns:
        st.error("Backtest returned no equity curve.")
        st.stop()

    status.success("Backtest complete ✅")

    st.subheader("Equity Curve")
    eq = daily.set_index("Date")["Equity"]
    st.line_chart(eq)

    daily_ret = daily["DailyReturn"] if "DailyReturn" in daily.columns else eq.pct_change().fillna(0.0)
    perf = {
        "Final": float(eq.iloc[-1]),
        "CAGR": _cagr(eq),
        "Sharpe": _sharpe(daily_ret, rf_daily=float(cash_daily_return)),
        "MaxDD": _maxdd(eq),
        "Days": int(len(eq)),
        "Tickers": ", ".join(tickers),
    }

    st.subheader("Summary")
    st.dataframe(
        pd.DataFrame([perf]).style.format({
            "Final": "{:,.2f}",
            "CAGR": "{:.2%}",
            "Sharpe": "{:.2f}",
            "MaxDD": "{:.2%}",
        })
    )
else:
    st.info("Configure options in the sidebar, then click **Run Backtest**.")