"""Download and cache historical price data for a configurable universe of tickers."""

import os

import pandas as pd
import yfinance as yf


START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "XOM", "JNJ", "SPY"]
DATA_DIR = "data"


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted close prices, with a CSV cache to avoid repeated downloads."""
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, f"prices_{start}_{end}.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    prices = prices[tickers].dropna(how="all")
    prices.to_csv(cache_path)
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns, forward-filled only within short gaps."""
    prices = prices.ffill(limit=5).dropna(how="any")
    return prices.pct_change().dropna(how="any")
