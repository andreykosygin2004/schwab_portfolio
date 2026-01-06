import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from analytics import normalize_transactions

prices = pd.read_csv("data/historical_prices.csv", parse_dates=["Date"]).set_index("Date").sort_index()
print(prices.index.min(), prices.index.max())
print(prices.index[:5])

tx = pd.read_csv("data/schwab_transactions.csv")
tx = normalize_transactions(tx)

print("min Date:", tx["Date"].min())
print("first 5 MoneyLink Transfers:")
print(tx[tx["Action"].str.strip()=="MoneyLink Transfer"][["Date","Action","Amount","Description","Symbol","Quantity"]].head(5))