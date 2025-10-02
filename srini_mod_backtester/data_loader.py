from __future__ import annotations
import datetime as dt
import pandas as pd
import yfinance as yf


def _tidy_from_wide(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalize yfinance wide dataframe to tidy OHLCV with Ticker column."""
    frames = []
    if len(tickers) == 1:
        sub = df.copy()
        sub.columns = [c.title() for c in sub.columns]
        sub["Ticker"] = tickers[0]
        frames.append(sub.reset_index().rename(columns={"Date": "Date"}))
    else:
        for t in tickers:
            sub = df[t].copy()
            sub.columns = [c.title() for c in sub.columns]
            sub["Ticker"] = t
            frames.append(sub.reset_index().rename(columns={"Date": "Date"}))
    out = pd.concat(frames, ignore_index=True)
    cols = ["Ticker", "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    out = out[cols]
    out = out.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return out


def get_prices(
    tickers: list[str] | str,
    start: str | dt.date | None = "2005-01-01",
    end: str | dt.date | None = None,
    source: str = "yahoo",  # "yahoo" | "stooq"
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Returns tidy OHLCV DataFrame with columns:
    ['Ticker','Date','Open','High','Low','Close','Adj Close','Volume']
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    if end is None:
        end = dt.date.today().isoformat()

    if source.lower() == "yahoo":
        df = yf.download(
            tickers=" ".join(tickers),
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        if df.empty:
            raise ValueError("No data downloaded from Yahoo. Check tickers or date range.")
        return _tidy_from_wide(df, tickers)

    elif source.lower() == "stooq":
        frames = []
        for t in tickers:
            stooq_t = f"{t.lower()}.us"  # Stooq US suffix
            sub = yf.download(
                stooq_t,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
                progress=False,
            )
            if sub.empty:
                continue
            sub.columns = [c.title() for c in sub.columns]
            sub["Ticker"] = t
            frames.append(sub.reset_index())
        if not frames:
            raise ValueError("No data downloaded from Stooq for provided tickers.")
        out = pd.concat(frames, ignore_index=True).rename(columns={"Date": "Date"})
        cols = ["Ticker", "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        out = out[cols].sort_values(["Ticker", "Date"]).reset_index(drop=True)
        return out

    else:
        raise ValueError("source must be 'yahoo' or 'stooq'")