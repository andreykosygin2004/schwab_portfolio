import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import datetime

# My Schwab Transaction Data
transactions = pd.read_csv("data/schwab_transactions.csv", skip_blank_lines=True)
unique_tickers = transactions["Symbol"].dropna().astype(str).unique().tolist()

price_data = yf.download(unique_tickers, start="2010-01-01", auto_adjust=True)["Close"]
price_data_unadjusted = yf.download(unique_tickers, start="2010-01-01", auto_adjust=False)
price_data_unadjusted.to_csv("data/historical_prices_unadjusted.csv")

# Benchmarks Data
benchmarks = ["^GSPC", "^IXIC"]
benchmark_data = yf.download(benchmarks, start="2010-01-01", auto_adjust=True)["Close"]

# Government Data
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime.today()

treasury_10y = web.DataReader("DGS10", "fred", start, end)
cpi = web.DataReader("CPIAUCSL", "fred", start, end)

# Volatility Data
vix = yf.download("^VIX", start="2010-01-01", auto_adjust=True)["Close"]