from __future__ import annotations

import numpy as np
import pandas as pd


def align_series(*series: pd.Series) -> list[pd.Series]:
    idx = None
    for s in series:
        if s is None or s.empty:
            return [pd.Series(dtype=float) for _ in series]
        idx = s.index if idx is None else idx.intersection(s.index)
    return [s.reindex(idx).dropna() for s in series]


def annualize_return_cagr(returns: pd.Series, periods_per_year: int) -> float:
    if returns.empty:
        return np.nan
    total = (1 + returns).prod()
    return total ** (periods_per_year / len(returns)) - 1


def annualize_vol(returns: pd.Series, periods_per_year: int) -> float:
    if returns.empty:
        return np.nan
    return float(returns.std() * np.sqrt(periods_per_year))
