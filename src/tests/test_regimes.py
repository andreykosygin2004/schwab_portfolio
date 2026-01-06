import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analytics.regimes import label_regimes, simulate_overlay


def test_label_regimes_priority():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    features = pd.DataFrame({
        "spy_mom_63": [0.05, 0.05, 0.05],
        "spy_vol_20": [0.03, 0.01, 0.01],
        "hyg_mom_21": [-0.04, 0.01, 0.01],
        "hyg_dd": [-0.09, -0.01, -0.01],
        "tlt_dd": [-0.07, -0.02, -0.02],
        "uso_mom_21": [0.0, 0.09, 0.09],
        "tip_mom_21": [0.0, -0.03, -0.03],
    }, index=idx)
    labels = label_regimes(features, "Balanced")
    assert labels.iloc[0] == "Risk-Off / Credit Stress"
    assert labels.iloc[1] == "Inflation Shock"
    assert labels.iloc[2] == "Inflation Shock"


def test_simulate_overlay_alignment():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    port_ret = pd.Series([0.01, -0.01, 0.02], index=idx)
    labels = pd.Series(["Risk-On", "Neutral / Transition", "Risk-Off / Credit Stress"], index=idx)
    exposure_map = {"Risk-On": 1.0, "Neutral / Transition": 0.8, "Risk-Off / Credit Stress": 0.4}
    overlay, exposure = simulate_overlay(port_ret, labels, exposure_map)
    assert len(overlay) == len(port_ret)
    assert np.isclose(exposure.iloc[-1], 0.4)
