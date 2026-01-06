import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analytics.factors import align_returns, fit_ols


def test_align_returns_intersection():
    idx1 = pd.date_range("2020-01-01", periods=5, freq="D")
    idx2 = pd.date_range("2020-01-03", periods=5, freq="D")
    y = pd.Series(np.arange(5), index=idx1)
    X = pd.DataFrame({"A": np.arange(5)}, index=idx2)
    y_aligned, X_aligned = align_returns(y, X)
    assert y_aligned.index.min() == idx2[0]
    assert len(y_aligned) == len(X_aligned)


def test_fit_ols_beta():
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    x = pd.Series(np.linspace(0, 1, 20), index=idx)
    y = 2.0 * x + 0.01
    X = pd.DataFrame({"factor": x})
    res = fit_ols(y, X)
    beta = res["betas"]["factor"]
    assert np.isclose(beta, 2.0, atol=1e-2)
