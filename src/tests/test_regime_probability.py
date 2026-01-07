import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analytics.regime_probability import make_labels, split_by_time, SplitConfig


def test_make_labels_horizon():
    idx = pd.date_range("2024-01-01", periods=6, freq="W-FRI")
    regimes = pd.Series(
        ["Neutral / Transition", "Risk-Off / Credit Stress", "Neutral / Transition", "Neutral / Transition", "Risk-Off / Credit Stress", "Neutral / Transition"],
        index=idx,
    )
    y = make_labels(regimes, 2)
    assert y.iloc[0] == 1
    assert y.iloc[1] == 0


def test_split_integrity():
    idx = pd.date_range("2020-01-03", periods=10, freq="W-FRI")
    X = pd.DataFrame({"a": range(10)}, index=idx)
    y = pd.Series([0, 1] * 5, index=idx)
    splits = SplitConfig(
        train_end=idx[4],
        val_end=idx[6],
        test_start=idx[7],
        test_end=idx[9],
    )
    split = split_by_time(X, y, splits)
    assert split["X_train"].index.max() <= splits.train_end
    assert split["X_test"].index.min() >= splits.test_start
