# streamlit_app.py  — minimal, robust data-loader front-end
from __future__ import annotations

import streamlit as st
import pandas as pd

# import the loader from your package
from srini_mod_backtester.data_loader import load_adj_close

# optional: if your backtest module is ready, you can wire it later
# from srini_mod_backtester.backtest_core import backtest_one

# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="US Backtester (Srini) · Modular", layout="wide")
    st.title("US Backtester (Modular)")

    # ------------------- Sidebar Inputs --------------------------------------
    with st.sidebar:
        st.header("Settings")
        tickers = st.text_input("Tickers (comma-separated)", value="SPY,XLK,ACN")
        start   = st.text_input("Start date (YYYY-MM-DD)", value="2015-01-01")
        end     = st.text_input("End date (YYYY-MM-DD)",   value="2025-10-02")

        st.caption("Click **Load prices** to fetch with Yahoo ➜ Stooq fallback.")
        go = st.button("Load prices", type="primary")

    # ------------------- Load & Diagnose -------------------------------------
    if go:
        tlist = [t.strip() for t in tickers.split(",") if t.strip()]
        if not tlist:
            st.error("Please enter at least one ticker.")
            st.stop()

        with st.spinner("Downloading (Yahoo → Stooq fallback)…"):
            data, failed, notes = load_adj_close(tlist, start, end)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Source notes")
            st.json(notes)

        with c2:
            st.subheader("Failed")
            st.write(", ".join(failed) or "None")

        with c3:
            st.subheader("Got data for")
            got = [k for k, v in data.items() if isinstance(v, pd.DataFrame) and not v.empty]
            st.write(", ".join(got) or "None")

        if not got:
            st.error("No data downloaded. Check tickers or date range.")
            st.stop()

        # --------------- Quick preview of the first successful ticker ---------
        t0 = got[0]
        st.markdown(f"### Preview: `{t0}` (first 10 rows)")
        st.dataframe(data[t0].head(10))

        # --------------- (Placeholder) Run strategy section -------------------
        st.divider()
        st.info(
            "Prices loaded successfully. Next step: re-attach your strategies "
            "(SMA/RSI/Composite), add Bollinger Bands, capital simulation, and "
            "export buttons. We’ll hook those in once this page is clean."
        )

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()