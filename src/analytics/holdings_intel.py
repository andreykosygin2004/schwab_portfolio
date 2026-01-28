from __future__ import annotations

import numpy as np
import pandas as pd


def concentration_metrics(weights: pd.Series) -> dict:
    weights = weights.dropna()
    if weights.empty:
        return {"hhi": np.nan, "top1": np.nan, "top3": np.nan}
    weights = weights / weights.sum()
    hhi = float((weights ** 2).sum())
    top1 = float(weights.nlargest(1).sum())
    top3 = float(weights.nlargest(3).sum())
    return {"hhi": hhi, "top1": top1, "top3": top3}


def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov()


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    if weights.empty or cov.empty:
        return pd.DataFrame()
    weights = weights.reindex(cov.columns).fillna(0.0)
    port_var = float(weights.T @ cov.values @ weights)
    if port_var <= 0:
        return pd.DataFrame()
    mcr = cov.values @ weights
    pcr = weights * mcr / port_var
    return pd.DataFrame({
        "weight": weights,
        "mcr": mcr,
        "pcr": pcr,
    }, index=cov.columns)


def scenario_impact(weights: pd.Series, shocks: dict[str, float]) -> float:
    if weights.empty or not shocks:
        return 0.0
    weights = weights / weights.sum()
    impact = 0.0
    for asset, shock in shocks.items():
        if asset in weights.index:
            impact += float(weights[asset]) * float(shock)
    return impact
