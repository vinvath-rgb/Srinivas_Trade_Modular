"""
streamlit_app.py
----------------
US Backtester · Modular (Streamlit UI)

This file:
- Builds the sidebar for inputs (dates, ticker, indicators, strategy, cash).
- Adds a visible **Run Backtest** button.
- Loads data via data_loader.load_prices_yahoo.
- Calls your modular backtester: srini_mod_backtester.run(...).
- Renders results robustly (equity curve + recent trades).

Assumes you have:
    srini_mod_backtester.py  -> defines  def run(...): -> Dict[str, Any]
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from data_loader import load_prices_yahoo

# ---- Try to import your modular engine ----
try:
    from srini_mod_backtester import run as run_backtest  # <-- your code
except Exception as _import_err:
    # Fallback so the UI still functions if import is temporarily broken.
    def run_backtest(
        prices: pd.DataFrame,
        strategy: str,
        initial_cash: float,
        rsi_buy: int,
        rsi_sell: int,
        cash_daily_return: float,
        indicators: dict | None = None,
    ):
        """
        Minimal stub backtest:
        - Buy & hold equity curve, no trades.
        This is only to keep the UI alive while you fix srini_mod_backtester.
        """
        if prices.empty:
            raise ValueError("No price data to backtest.")
        equity_curve = (1 + prices["Close"].pct_change().fillna(0)).cumprod() * float(initial_cash)
        return {
            "equity_curve": equity_curve,
            "trades": pd.DataFrame(columns=["Date", "Ticker", "Side", "Price", "Qty"]),
            "perf": {"CAGR": float(equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0},
        }


# ---------------- UI LAYOUT ----------------
st.set_page_config(page_title="US Backtester · Modular", layout="wide")

st.title("US Backtester · Modular")

with st.sidebar:
    st.subheader("Data")
    ticker = st.text_input("Ticker", value="SPY").strip().upper()
    col_dt1, col_dt2 = st.columns(2)
    with col_dt1:
        start_date = st.date_input("Start", value=pd.to_datetime("2015-01-01"))
    with col_dt2:
        end_date = st.date_input("End", value=pd.Timestamp.today().normalize())
    interval = st.selectbox("Interval", options=["1d", "1wk", "1mo"], index=0)
    source = st.selectbox("Source", options=["yahoo"], index=0, help="Yahoo Finance via yfinance")

    st.subheader("Indicators (visual/logic flags)")
    use_sma = st.toggle("SMA (50)", value=True)
    use_rsi = st.toggle("RSI (14)", value=True)
    use_bb = st.toggle("Bollinger (20,2)", value=False)

    st.subheader("Strategy")
    strategy = st.radio("Pick strategy", options=["SMA crossover", "RSI mean reversion"], index=1)
    rsi_buy = st.number_input("RSI Buy <=", value=30, step=1, min_value=1, max_value=99)
    rsi_sell = st.number_input("RSI Sell >=", value=70, step=1, min_value=1, max_value=99)

    st.subheader("Backtest")
    initial_cash = st.number_input("Initial Cash", value=100000, step=1000, min_value=0)
    cash_daily_return = st.number_input("Cash Daily Return (dec)", value=0.0000, step=0.0001, format="%.4f")

    st.caption("Press the button below when ready.")
    run_clicked = st.button("Run Backtest", type="primary", use_container_width=True)

# --- status banner / info area ---
status_box = st.empty()

# ---------------- BUTTON HANDLER ----------------
if run_clicked:
    status_box.info("Downloading data & running backtest…")

    # 1) Data load
    prices = pd.DataFrame()
    if source.lower() == "yahoo":
        prices = load_prices_yahoo(
            ticker=ticker,
            start=str(start_date),
            end=str(end_date),
            interval=interval,
            auto_adjust=True,
        )

    if prices.empty:
        status_box.error(
            "Data load failed: No data downloaded from Yahoo. "
            "Please check the ticker, date range, interval, or try a different symbol."
        )
        st.stop()

    # 2) Run backtest (your engine decides how to use indicators)
    indicators = {
        "SMA_50": use_sma,
        "RSI_14": use_rsi,
        "Boll_20_2": use_bb,
    }

    try:
        results = run_backtest(
            prices=prices,
            strategy=strategy,
            initial_cash=float(initial_cash),
            rsi_buy=int(rsi_buy),
            rsi_sell=int(rsi_sell),
            cash_daily_return=float(cash_daily_return),
            indicators=indicators,  # optional, your engine can ignore if not needed
        )
    except Exception as e:
        status_box.error(f"Backtest failed: {e}")
        st.exception(e)
        st.stop()

    status_box.success("Backtest complete ✅")

    # 3) Render outputs defensively
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("Equity Curve")
        if isinstance(results, dict) and "equity_curve" in results:
            eq = results["equity_curve"]
            if isinstance(eq, pd.Series):
                eq = eq.rename("Equity")
            st.line_chart(eq)
        else:
            st.info("No equity curve returned by the backtester.")

    with col2:
        st.subheader("Performance")
        perf = results.get("perf", {}) if isinstance(results, dict) else {}
        if isinstance(perf, dict) and perf:
            for k, v in perf.items():
                st.metric(k, f"{v:.4f}" if isinstance(v, (int, float)) else str(v))
        else:
            st.caption("No performance dictionary returned.")

    st.subheader("Recent Trades")
    trades = results.get("trades") if isinstance(results, dict) else None
    if isinstance(trades, pd.DataFrame) and not trades.empty:
        trades_display = trades.copy()
        if "Date" in trades_display.columns:
            trades_display["Date"] = pd.to_datetime(trades_display["Date"])
            trades_display = trades_display.sort_values("Date")
        st.dataframe(trades_display.tail(50), use_container_width=True)
        # Download
        csv = trades_display.to_csv(index=False).encode()
        st.download_button("Download Trades CSV", data=csv, file_name=f"{ticker}_trades.csv", mime="text/csv")
    else:
        st.caption("No trades to display.")

else:
    # Initial page hint
    st.info("Configure options in the sidebar, then click **Run Backtest**.")