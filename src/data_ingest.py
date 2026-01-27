import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import datetime

# My Schwab Transaction Data
transactions = pd.read_csv("data/schwab_transactions.csv", skip_blank_lines=True)
unique_tickers = transactions["Symbol"].dropna().astype(str).unique().tolist()
IGNORE_TICKERS = {"35952H601"}
unique_tickers = [t for t in unique_tickers if t not in IGNORE_TICKERS]

price_data = yf.download(unique_tickers, start="2010-01-01", auto_adjust=True)["Close"]

actions = yf.download(
    unique_tickers,
    start="2010-01-01",
    actions=True
)
stock_splits = actions["Stock Splits"]
stock_splits_long = (
    stock_splits
    .stack()
    .reset_index()
    .rename(columns={
        "Date": "date",
        "level_1": "symbol",
        0: "split_ratio"
    })
)
stock_splits_long = stock_splits_long[stock_splits_long["split_ratio"] != 0].sort_values("date")
stock_splits_long.to_csv("data/stock_splits.csv", index=False)

# Benchmarks Data
benchmarks = ["^GSPC", "^IXIC"]
benchmark_data = yf.download(benchmarks, start="2010-01-01", auto_adjust=True)["Close"]
benchmark_data = benchmark_data.rename(columns={
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ Composite"
})

# Government Data
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime.today()

treasury_10y = web.DataReader("DGS10", "fred", start, end)
cpi = web.DataReader("CPIAUCSL", "fred", start, end)

# Volatility Data
vix = yf.download("^VIX", start="2010-01-01", auto_adjust=True)["Close"]