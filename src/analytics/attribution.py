from __future__ import annotations

import numpy as np
import pandas as pd


def compute_contributions(weights: pd.Series, returns: pd.Series) -> pd.Series:
    weights = weights.reindex(returns.index).fillna(0.0)
    return weights * returns


def top_contributors(contrib: pd.Series, n: int) -> pd.Series:
    if contrib.empty:
        return contrib
    return contrib.reindex(contrib.abs().sort_values(ascending=False).head(n).index)


def factor_period_contributions(betas: pd.Series, factor_returns: pd.DataFrame) -> pd.DataFrame:
    if betas.empty or factor_returns.empty:
        return pd.DataFrame()
    aligned = factor_returns.reindex(columns=betas.index).dropna(how="all")
    return aligned.multiply(betas, axis=1)


def build_pm_memo(metrics: dict) -> list[str]:
    memo = []
    if metrics.get("top_contrib"):
        memo.append(f"Top contributor: {metrics['top_contrib']}.")
    if metrics.get("top_detractor"):
        memo.append(f"Largest detractor: {metrics['top_detractor']}.")
    if metrics.get("top_factor"):
        memo.append(f"Dominant factor: {metrics['top_factor']}.")
    if metrics.get("residual"):
        memo.append(f"Residual return over window: {metrics['residual']}.")
    if metrics.get("regime_note"):
        memo.append(metrics["regime_note"])
    return memo
