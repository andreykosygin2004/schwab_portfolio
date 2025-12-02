import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

prices = pd.read_csv("data/historical_prices.csv", parse_dates=True, index_col=0)
holdings = pd.read_csv("data/holdings_timeseries.csv", parse_dates=True, index_col=0)

nvda_series = prices["NFLX"]
nvda_series_2 = holdings["MV_NFLX"]

plt.figure(figsize=(10, 5))
plt.plot(nvda_series_2)
plt.title("NFLX Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(nvda_series)
plt.title("FCEL Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()

