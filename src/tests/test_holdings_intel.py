import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analytics.holdings_intel import concentration_metrics, risk_contributions


def test_concentration_metrics():
    weights = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
    metrics = concentration_metrics(weights)
    assert np.isclose(metrics["hhi"], 0.38)
    assert np.isclose(metrics["top1"], 0.5)
    assert np.isclose(metrics["top3"], 1.0)


def test_risk_contributions_simple():
    weights = pd.Series([0.6, 0.4], index=["A", "B"])
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"])
    rc = risk_contributions(weights, cov)
    assert not rc.empty
    assert np.isclose(rc["pcr"].sum(), 1.0)
