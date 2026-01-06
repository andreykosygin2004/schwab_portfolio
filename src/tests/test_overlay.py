import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analytics.overlay import apply_overlay, compute_overlay_weights


def test_apply_overlay_basic():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    port_ret = pd.Series([0.01, 0.0, -0.01], index=idx)
    qqq_ret = pd.Series([0.01, 0.01, 0.01], index=idx)
    weights = pd.DataFrame({"w_qqq": [1.0, 1.0, 1.0], "w_tlt": 0.0, "w_gld": 0.0}, index=idx)
    overlay = apply_overlay(port_ret, qqq_ret, None, None, weights)
    expected = port_ret - qqq_ret
    assert np.allclose(overlay.values, expected.values)


def test_overlay_constraints():
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    labels = pd.Series(["Risk-Off / Credit Stress", "Risk-Off / Credit Stress"], index=idx)
    beta = pd.Series([2.0, 2.0], index=idx)
    config = {
        "target_beta_mult": {"Risk-Off / Credit Stress": 0.2},
        "tlt_tilt": {"Risk-Off / Credit Stress": 0.5},
        "gld_tilt": {"Risk-Off / Credit Stress": 0.5},
        "max_hedge": 1.0,
        "gross_cap": 1.0,
        "allow_leverage": False,
    }
    weights = compute_overlay_weights(labels, beta, config)
    assert (weights.abs().sum(axis=1) <= 1.0 + 1e-6).all()
