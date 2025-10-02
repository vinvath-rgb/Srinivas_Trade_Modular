import pandas as pd
import streamlit as st

from srini_mod_backtester.data_loader import get_prices
from srini_mod_backtester.indicators import add_sma, add_rsi, add_bbands
from srini_mod_backtester.strategies import (
    sma_crossover_signals,
    rsi_mean_reversion,
)
from srini_mod_backtester.backtest_core import equity_curve_long_only

st.set_page_config(page_title="US Backtester (Modular)", layout="wide")

st.title("US Backtester · Modular")

with st.sidebar:
    st.subheader("Data")
    tickers = st.text_input("Tickers (comma-separated)", value="SPY, XLK, ACN")
    tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start", value=pd.to_datetime("2015-01-01").date())
    with col2:
        end = st.date_input("End", value=pd.to_datetime("today").date())

    source = st.selectbox("Source", ["yahoo", "stooq"], index=0)
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

    st.subheader("Indicators")
    use_sma = st.toggle("SMA (50)", value=True)
    use_rsi = st.toggle("RSI (14)", value=True)
    use_bb = st.toggle("Bollinger (20,2)", value=False)

    st.subheader("Strategy")
    strategy = st.radio("Pick strategy", ["SMA crossover", "RSI mean reversion"], index=0)
    if strategy == "SMA crossover":
        fast = st.number_input("Fast SMA", min_value=5, max_value=200, value=20, step=1)
        slow = st.number_input("Slow SMA", min_value=10, max_value=400, value=50, step=1)
    else:
        low = st.number_input("RSI Buy <", min_value=5, max_value=50, value=30, step=1)
        high = st.number_input("RSI Sell >", min_value=50, max_value=95, value=70, step=1)

    st.subheader("Backtest")
    init_cash = st.number_input("Initial Cash", min_value=1_000, max_value=10_000_000, value=100_000, step=1_000)
    cash_r = st.number_input("Cash Daily Return (dec)", min_value=0.0, max_value=0.01, value=0.0, step=0.0001, format="%.4f")

st.success("Configured. Load + compute below 👇")

@st.cache_data(show_spinner=False)
def load_data(_tickers, _start, _end, _source, _interval):
    return get_prices(_tickers, start=_start, end=_end, source=_source, interval=_interval)

# Load
try:
    df = load_data(tickers, start.isoformat(), end.isoformat(), source, interval)
    st.toast(f"Loaded {df['Ticker'].nunique()} tickers · {len(df):,} rows", icon="✅")
except Exception as e:
    st.error(f"Data load failed: {e}")
    st.stop()

# Indicators
if use_sma:
    df = add_sma(df, window=50)
if use_rsi:
    df = add_rsi(df, window=14)
if use_bb:
    df = add_bbands(df, window=20, num_std=2)

# Strategy -> Signal
if strategy == "SMA crossover":
    df = sma_crossover_signals(df, fast=fast, slow=slow)
else:
    if "RSI14" not in df.columns:
        df = add_rsi(df, window=14)
    df = rsi_mean_reversion(df, low=low, high=high)

# Equity curve
curve = equity_curve_long_only(df, init_cash=init_cash, allow_cash_return=cash_r)

# Layout
colA, colB = st.columns([1, 1])
with colA:
    st.subheader("Sample Data")
    st.dataframe(df.sort_values(["Date", "Ticker"]).tail(500), use_container_width=True)
    st.download_button(
        "Download Data CSV",
        data=df.to_csv(index=False),
        file_name="backtest_view.csv",
        mime="text/csv",
    )

with colB:
    st.subheader("Equity Curve")
    st.line_chart(curve.set_index("Date")["Equity"])
    st.caption(
        f"Final Equity: **${curve['Equity'].iloc[-1]:,.2f}** | Max rows: {len(curve):,}"
    )

st.divider()
st.write("**Notes**: Positions use *yesterday's* signal to avoid lookahead. Equal weight among active longs per day. If no signals, equity sits in cash with optional daily cash return.")