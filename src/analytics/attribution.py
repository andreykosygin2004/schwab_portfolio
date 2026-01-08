from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.common import time_varying_weights


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


def time_series_attribution(
    mv_df: pd.DataFrame,
    price_df: pd.DataFrame,
    total_value: pd.Series,
) -> pd.DataFrame:
    if mv_df.empty or price_df.empty or total_value.empty:
        return pd.DataFrame()

    weights_lag = time_varying_weights(mv_df, freq="Daily", total_value=total_value)

    prices = price_df.reindex(weights.index).ffill().dropna(how="all")
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    aligned = weights_lag.index.intersection(returns.index)
    weights_lag = weights_lag.reindex(aligned).fillna(0.0)
    returns = returns.reindex(aligned)

    contrib = weights_lag * returns
    total_contrib = contrib.sum(axis=0)
    avg_weight = weights_lag.mean(axis=0)
    total_return = (1 + returns).prod() - 1

    out = pd.DataFrame({
        "avg_weight": avg_weight,
        "total_return": total_return,
        "contribution": total_contrib,
    })
    out["pct_total"] = out["contribution"] / out["contribution"].sum() if out["contribution"].sum() != 0 else 0.0
    return out.sort_values("contribution", ascending=False)
