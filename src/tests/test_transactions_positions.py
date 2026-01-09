import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.transactions import build_positions_from_transactions


def test_sell_rebuy_positions():
    dates = pd.date_range("2024-01-02", periods=10, freq="D")
    prices = pd.DataFrame({"NVDA": [100 + i for i in range(len(dates))]}, index=dates)
    tx = pd.DataFrame(
        [
            {"Date": "2024-01-03", "Action": "Buy", "Symbol": "NVDA", "Quantity": 10, "Price": 101, "Amount": -1010},
            {"Date": "2024-01-06", "Action": "Sell", "Symbol": "NVDA", "Quantity": 10, "Price": 104, "Amount": 1040},
            {"Date": "2024-01-08", "Action": "Buy", "Symbol": "NVDA", "Quantity": 5, "Price": 106, "Amount": -530},
        ]
    )
    positions, debug = build_positions_from_transactions(tx, prices, dates.min(), dates.max())
    assert not positions.empty
    assert positions.loc[pd.Timestamp("2024-01-04"), "NVDA"] == 10
    assert positions.loc[pd.Timestamp("2024-01-07"), "NVDA"] == 0
    assert positions.loc[pd.Timestamp("2024-01-09"), "NVDA"] == 5
