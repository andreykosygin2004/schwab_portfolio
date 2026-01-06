from __future__ import annotations

import numpy as np
import pandas as pd


def align_returns(port_ret: pd.Series, factor_ret: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    if port_ret.empty or factor_ret.empty:
        return pd.Series(dtype=float), pd.DataFrame()
    idx = port_ret.index.intersection(factor_ret.index)
    return port_ret.reindex(idx), factor_ret.reindex(idx)


def fit_ols(y: pd.Series, X: pd.DataFrame) -> dict:
    if y.empty or X.empty:
        return {"alpha": np.nan, "betas": pd.Series(dtype=float), "r2": np.nan, "stderr": None}
    X = X.copy()
    X["alpha"] = 1.0
    cols = ["alpha"] + [c for c in X.columns if c != "alpha"]
    X = X[cols]
    y_vals = y.to_numpy()
    X_vals = X.to_numpy()
    beta, *_ = np.linalg.lstsq(X_vals, y_vals, rcond=None)
    y_hat = X_vals @ beta
    ss_res = np.sum((y_vals - y_hat) ** 2)
    ss_tot = np.sum((y_vals - y_vals.mean()) ** 2)
    r2 = 0.0 if ss_tot == 0 else 1 - (ss_res / ss_tot)

    stderr = None
    n, k = X_vals.shape
    if n > k:
        sigma2 = ss_res / (n - k)
        cov = sigma2 * np.linalg.inv(X_vals.T @ X_vals)
        stderr = np.sqrt(np.diag(cov))

    alpha = beta[0]
    betas = pd.Series(beta[1:], index=[c for c in X.columns if c != "alpha"])
    return {"alpha": alpha, "betas": betas, "r2": r2, "stderr": stderr}


def rolling_multifactor(y: pd.Series, X: pd.DataFrame, window: int) -> tuple[pd.DataFrame, pd.Series]:
    if y.empty or X.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    betas = []
    r2s = []
    dates = []
    for i in range(window, len(y) + 1):
        y_win = y.iloc[i - window:i]
        X_win = X.iloc[i - window:i]
        res = fit_ols(y_win, X_win)
        betas.append(res["betas"])
        r2s.append(res["r2"])
        dates.append(y_win.index[-1])
    beta_df = pd.DataFrame(betas, index=dates)
    r2_series = pd.Series(r2s, index=dates)
    return beta_df, r2_series


def factor_contributions(betas: pd.DataFrame, factor_returns: pd.DataFrame) -> pd.DataFrame:
    if betas.empty or factor_returns.empty:
        return pd.DataFrame()
    aligned = betas.index.intersection(factor_returns.index)
    betas = betas.reindex(aligned).ffill()
    factor_returns = factor_returns.reindex(aligned)
    return betas * factor_returns
