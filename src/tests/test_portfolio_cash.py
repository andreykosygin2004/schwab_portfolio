import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analytics.portfolio import accrue_cash_balance
from analytics import compute_clean_cash_balance


def test_cash_accrual_positive_rf():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    cash_balance = pd.Series([100.0, 100.0, 100.0], index=idx)
    rf = pd.Series([0.0, 0.01, 0.01], index=idx)
    cash_value = accrue_cash_balance(cash_balance, rf)
    assert cash_value.iloc[-1] > cash_balance.iloc[-1]


def test_cash_accrual_equity_higher():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    cash_balance = pd.Series([100.0, 100.0, 100.0], index=idx)
    rf = pd.Series([0.0, 0.01, 0.01], index=idx)
    cash_value = accrue_cash_balance(cash_balance, rf)
    portfolio_value = pd.Series([1000.0, 1005.0, 1010.0], index=idx)
    old_equity = portfolio_value + cash_balance
    new_equity = portfolio_value + cash_value
    assert new_equity.iloc[-1] > old_equity.iloc[-1]


def test_cash_accrual_zero_rf():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    cash_balance = pd.Series([100.0, 120.0, 120.0], index=idx)
    rf = pd.Series([0.0, 0.0, 0.0], index=idx)
    cash_value = accrue_cash_balance(cash_balance, rf)
    assert np.isclose(cash_value.iloc[-1], 120.0)


def test_negative_moneylink_excluded():
    tx = pd.DataFrame({
        "Date": ["2020-01-01", "2020-01-02"],
        "Action": ["MoneyLink Transfer", "MoneyLink Transfer"],
        "Amount": [1000.0, -500.0],
        "Description": ["deposit", "withdrawal"],
        "Symbol": [np.nan, np.nan],
    })
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    cash_clean = compute_clean_cash_balance(tx, idx)
    assert cash_clean.iloc[-1] == 1000.0
