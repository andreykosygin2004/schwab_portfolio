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


def time_varying_weights(
    mv_df: pd.DataFrame,
    freq: str = "Daily",
    total_value: pd.Series | None = None,
) -> pd.DataFrame:
    if mv_df.empty:
        return pd.DataFrame()
    mv_df = mv_df.copy()
    mv_df.columns = [c.replace("MV_", "") for c in mv_df.columns]
    mv_df = mv_df.ffill().fillna(0.0)
    if freq == "Weekly":
        mv_df = mv_df.resample("W-FRI").last()
    if total_value is not None:
        total_value = total_value.reindex(mv_df.index).ffill()
        total_mv = total_value.replace(0, np.nan)
    else:
        total_mv = mv_df.sum(axis=1).replace(0, np.nan)
    weights = mv_df.div(total_mv, axis=0).fillna(0.0)
    weights_lag = weights.shift(1).fillna(0.0)
    return weights_lag
