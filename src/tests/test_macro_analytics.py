import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analytics_macro import beta_and_r2, compute_returns, rolling_corr, rolling_vol


def test_beta_and_r2_perfect_fit():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    x = pd.Series([1, 2, 3, 4, 5], index=idx)
    y = 2 * x + 1
    beta, r2 = beta_and_r2(y, x)
    assert np.isclose(beta, 2.0)
    assert np.isclose(r2, 1.0)


def test_rolling_corr_perfect():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    a = pd.Series([1, 2, 3, 4, 5], index=idx)
    b = pd.Series([1, 2, 3, 4, 5], index=idx)
    corr = rolling_corr(a, b, window=3)
    assert np.isclose(corr.dropna().iloc[-1], 1.0)


def test_rolling_vol_zero():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    returns = pd.Series([0.01, 0.01, 0.01, 0.01], index=idx)
    vol = rolling_vol(returns, window=3)
    assert np.isclose(vol.dropna().iloc[-1], 0.0)


def test_compute_returns_weekly_resample():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    prices = pd.Series(np.linspace(100, 110, 10), index=idx)
    weekly = compute_returns(prices, freq="Weekly")
    assert len(weekly) <= len(prices)
