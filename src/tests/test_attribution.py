import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analytics.attribution import compute_contributions, top_contributors


def test_contributions_sum():
    weights = pd.Series({"A": 0.6, "B": 0.4})
    returns = pd.Series({"A": 0.1, "B": -0.05})
    contrib = compute_contributions(weights, returns)
    assert np.isclose(contrib.sum(), 0.6 * 0.1 + 0.4 * -0.05)


def test_top_contributors():
    contrib = pd.Series({"A": 0.2, "B": -0.1, "C": 0.05})
    top = top_contributors(contrib, 2)
    assert "A" in top.index
    assert "B" in top.index
