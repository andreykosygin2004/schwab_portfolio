import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from strategies.factor_rotation import RotationParams, backtest_rotation, ETF_UNIVERSE


def _make_prices() -> pd.DataFrame:
    idx = pd.date_range("2020-01-31", periods=36, freq="M")
    data = {}
    for i, t in enumerate(ETF_UNIVERSE):
        base = 100 + i * 5
        data[t] = base * (1 + 0.01 * (np.arange(len(idx)) + i))
    return pd.DataFrame(data, index=idx)


def test_weights_sum_to_one():
    prices = _make_prices()
    params = RotationParams(trend_filter=False, vol_adjust=False, smooth_lambda=1.0)
    results = backtest_rotation(prices, params, regime_labels=None)
    weights = results["weights"]
    sums = weights.sum(axis=1).round(6)
    assert (sums > 0.99).all()
    assert (sums < 1.01).all()


def test_no_nan_returns_after_warmup():
    prices = _make_prices()
    params = RotationParams(trend_filter=False, vol_adjust=False, smooth_lambda=1.0)
    results = backtest_rotation(prices, params, regime_labels=None)
    returns = results["returns"]
    assert returns.notna().all()
