import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
import pandas_datareader.data as web


def main() -> None:
    transactions_path = Path("data/schwab_transactions.csv")
    if not transactions_path.exists():
        raise FileNotFoundError("data/schwab_transactions.csv not found (local-only).")

    transactions = pd.read_csv(transactions_path, skip_blank_lines=True)
    unique_tickers = transactions["Symbol"].dropna().astype(str).unique().tolist()
    ignore_tickers = {"35952H601"}
    unique_tickers = [t for t in unique_tickers if t not in ignore_tickers]

    price_data = yf.download(unique_tickers, start="2010-01-01", auto_adjust=True)["Close"]
    price_data.to_csv("data/historical_prices.csv")

    actions = yf.download(unique_tickers, start="2010-01-01", actions=True)
    stock_splits = actions["Stock Splits"]
    stock_splits_long = (
        stock_splits
        .stack()
        .reset_index()
        .rename(columns={"Date": "date", "level_1": "symbol", 0: "split_ratio"})
    )
    stock_splits_long = stock_splits_long[stock_splits_long["split_ratio"] != 0].sort_values("date")
    stock_splits_long.to_csv("data/stock_splits.csv", index=False)

    benchmarks = ["^GSPC", "^IXIC"]
    benchmark_data = yf.download(benchmarks, start="2010-01-01", auto_adjust=True)["Close"]
    benchmark_data = benchmark_data.rename(columns={"^GSPC": "S&P 500", "^IXIC": "NASDAQ Composite"})
    benchmark_data.to_csv("data/benchmark_prices.csv")

    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime.today()
    treasury_10y = web.DataReader("DGS10", "fred", start, end)
    cpi = web.DataReader("CPIAUCSL", "fred", start, end)
    treasury_10y.to_csv("data/treasury.csv")
    cpi.to_csv("data/cpi.csv")

    vix = yf.download("^VIX", start="2010-01-01", auto_adjust=True)["Close"]
    vix.to_csv("data/vol.csv")


if __name__ == "__main__":
    main()
