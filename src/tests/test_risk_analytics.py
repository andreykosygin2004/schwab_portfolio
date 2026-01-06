import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analytics.risk import drawdown_episodes, drawdown_series, max_drawdown, var_cvar


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


def test_drawdown_episodes_recovered():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    prices = pd.Series([100, 110, 90, 115], index=idx)
    episodes = drawdown_episodes(prices)
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep["peak"] == idx[1]
    assert ep["trough"] == idx[2]
    assert ep["recovery"] == idx[3]
    assert ep["duration_to_trough"] == 1
    assert ep["recovery_time"] == 1


def test_drawdown_episodes_unrecovered():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    prices = pd.Series([100, 110, 90, 95], index=idx)
    episodes = drawdown_episodes(prices)
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep["recovery"] is None
    assert ep["total_duration"] == 2


def test_drawdown_depth_consistency():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.Series([100, 120, 80, 90, 130], index=idx)
    dd_min = drawdown_series(prices).min()
    episodes = drawdown_episodes(prices)
    ep_min = min((ep["depth"] for ep in episodes), default=np.nan)
    assert np.isclose(dd_min, ep_min)
