import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analytics.attribution import time_series_attribution


def test_time_series_attribution_sold_position():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    mv = pd.DataFrame({
        "MV_A": [100, 100, 0, 0],
        "MV_B": [0, 0, 100, 100],
    }, index=idx)
    prices = pd.DataFrame({
        "A": [100, 110, 110, 110],
        "B": [100, 100, 105, 110],
    }, index=idx)
    total = mv.sum(axis=1)
    attr = time_series_attribution(mv, prices, total)
    assert attr.loc["A", "contribution"] != 0
    assert attr.loc["B", "contribution"] != 0
