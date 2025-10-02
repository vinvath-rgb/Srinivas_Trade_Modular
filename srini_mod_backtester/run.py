import streamlit as st
import pandas as pd
from .data_loader import load_adj_close
from .backtest_core import backtest_one as run_backtest
from .excel_export import to_excel

def main():
    st.title("Srini Modular Backtester (Package)")
    ticker = st.text_input("Ticker", "SPY")
    c1, c2 = st.columns(2)
    start = c1.date_input("Start", value=pd.to_datetime("2015-01-01")).strftime("%Y-%m-%d")
    end   = c2.date_input("End",   value=pd.Timestamp.today()).strftime("%Y-%m-%d")
    strat = st.selectbox("Strategy", ["SMA", "RSI"])

    if strat == "SMA":
        f1, f2 = st.columns(2)
        fast = f1.number_input("Fast SMA", 2, 200, 20, 1)
        slow = f2.number_input("Slow SMA", 5, 400, 100, 5)
        params = {"fast": int(fast), "slow": int(slow)}
    else:
        r1, r2, r3 = st.columns(3)
        lb = r1.number_input("RSI lookback", 2, 100, 14, 1)
        buy = r2.number_input("RSI Buy <", 5, 50, 30, 1)
        sell = r3.number_input("RSI Sell >", 50, 95, 70, 1)
        params = {"lb": int(lb), "buy": int(buy), "sell": int(sell)}

    vol_target = st.slider("Vol target (ann.)", 0.05, 0.40, 0.15, 0.01)

    if st.button("Run"):
        px = load_adj_close(ticker, start, end)
        if px.empty:
            st.error("No data."); return
        equity, stats = run_backtest(px, strategy=strat, params=params, vol_target=vol_target)
        st.line_chart(equity)
        st.write(stats)
        x = to_excel({"Ticker": ticker, "Start": start, "End": end, "Strategy": strat, **params, "VolTarget": vol_target}, stats, equity)
        st.download_button("Download Excel", x, file_name=f"Backtest_{ticker}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
