from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.factors import fit_ols


def _align_returns(y: pd.Series, X: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    aligned = pd.concat([y, X], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float), pd.DataFrame()
    y_aligned = aligned.iloc[:, 0]
    X_aligned = aligned.iloc[:, 1:]
    return y_aligned, X_aligned


def fit_factor_model(y: pd.Series, X: pd.DataFrame) -> dict:
    y_aligned, X_aligned = _align_returns(y, X)
    if y_aligned.empty or X_aligned.empty:
        return {"alpha": np.nan, "betas": pd.Series(dtype=float), "residuals": pd.Series(dtype=float)}
    fit = fit_ols(y_aligned, X_aligned)
    alpha = fit["alpha"]
    betas = fit["betas"]
    explained = alpha + (X_aligned * betas).sum(axis=1)
    residuals = y_aligned - explained
    return {"alpha": alpha, "betas": betas, "residuals": residuals}


def fit_factor_model_rolling(y: pd.Series, X: pd.DataFrame, window: int) -> tuple[pd.Series, pd.DataFrame]:
    y_aligned, X_aligned = _align_returns(y, X)
    if y_aligned.empty or X_aligned.empty:
        return pd.Series(dtype=float), pd.DataFrame()
    residuals = pd.Series(index=y_aligned.index, dtype=float)
    betas = pd.DataFrame(index=y_aligned.index, columns=X_aligned.columns, dtype=float)
    for i in range(window - 1, len(y_aligned)):
        y_win = y_aligned.iloc[i - window + 1 : i + 1]
        X_win = X_aligned.iloc[i - window + 1 : i + 1]
        fit = fit_ols(y_win, X_win)
        alpha = fit["alpha"]
        beta = fit["betas"]
        pred = alpha + (X_aligned.iloc[i] * beta).sum()
        residuals.iloc[i] = y_aligned.iloc[i] - pred
        betas.iloc[i] = beta
    return residuals.dropna(), betas.dropna(how="all")


def compute_holding_alpha_contrib(weights: pd.DataFrame, residuals: pd.DataFrame) -> pd.DataFrame:
    if weights.empty or residuals.empty:
        return pd.DataFrame()
    aligned = weights.reindex(residuals.index).fillna(0.0)
    weights_lag = aligned.shift(1).fillna(0.0)
    contrib = weights_lag * residuals
    return contrib


def compute_portfolio_explained_residual(
    port_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    fit = fit_factor_model(port_returns, factor_returns)
    residuals = fit["residuals"]
    if residuals.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    aligned_factors = factor_returns.reindex(residuals.index)
    explained = fit["alpha"] + (aligned_factors * fit["betas"]).sum(axis=1)
    return explained, residuals
