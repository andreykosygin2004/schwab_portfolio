import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analytics.risk import drawdown_episodes, max_drawdown, var_cvar


def test_max_drawdown_simple():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    prices = pd.Series([100, 110, 90, 105], index=idx)
    dd = max_drawdown(prices)
    assert np.isclose(dd, -0.18181818)


def test_var_cvar_basic():
    returns = pd.Series([-0.05, 0.01, 0.02, -0.02, 0.03])
    var_95, cvar_95 = var_cvar(returns, alpha=0.05)
    assert var_95 <= 0
    assert cvar_95 <= var_95


def test_drawdown_episodes_single():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.Series([100, 95, 90, 92, 101], index=idx)
    episodes = drawdown_episodes(prices)
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep["start"] == idx[1]
    assert ep["trough"] == idx[2]
    assert ep["recovery"] == idx[4]
