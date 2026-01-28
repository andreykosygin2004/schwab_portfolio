from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.factor_rotation import apply_turnover_smoothing


def _apply_max_cap(weights: pd.Series, max_cap: float) -> pd.Series:
    capped = weights.clip(upper=max_cap)
    total = capped.sum()
    return capped / total if total > 0 else capped


def compute_correlation_aware_weights(
    base_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    core_returns: pd.Series,
    lookback_periods: int,
    gamma: float,
    use_abs_corr: bool,
    max_cap: float,
    smooth_lambda: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply a correlation penalty to base weights using only past returns.
    """
    if base_weights.empty or asset_returns.empty or core_returns.empty:
        return pd.DataFrame(), pd.DataFrame()

    weights = []
    diag_rows = []
    idx = base_weights.index.intersection(asset_returns.index).intersection(core_returns.index)
    base_weights = base_weights.reindex(idx).fillna(0.0)
    asset_returns = asset_returns.reindex(idx).fillna(0.0)
    core_returns = core_returns.reindex(idx).fillna(0.0)

    for dt in idx:
        window = asset_returns.loc[:dt].tail(int(lookback_periods))
        core_win = core_returns.loc[:dt].tail(int(lookback_periods))
        if window.empty or core_win.empty:
            weights.append(base_weights.loc[dt])
            continue
        corr = window.corrwith(core_win).fillna(0.0)
        metric = corr.abs() if use_abs_corr else corr.clip(lower=0.0)
        penalty = np.exp(-gamma * metric)
        adjusted = base_weights.loc[dt] * penalty
        if adjusted.sum() <= 0:
            adjusted = base_weights.loc[dt]
        adjusted = _apply_max_cap(adjusted, max_cap)
        weights.append(adjusted)
        for asset in adjusted.index:
            diag_rows.append({
                "date": dt,
                "asset": asset,
                "corr": float(corr.get(asset, 0.0)),
                "penalty": float(penalty.get(asset, 0.0)),
            })

    weights_df = pd.DataFrame(weights, index=idx, columns=base_weights.columns).fillna(0.0)
    weights_df = apply_turnover_smoothing(weights_df, smooth_lambda)
    diag_df = pd.DataFrame(diag_rows)
    return weights_df, diag_df


def backtest_with_weights(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    tc_bps: float,
) -> dict:
    if weights.empty or asset_returns.empty:
        return {}
    aligned = weights.index.intersection(asset_returns.index)
    weights = weights.reindex(aligned).fillna(0.0)
    asset_returns = asset_returns.reindex(aligned).fillna(0.0)
    weights_lag = weights.shift(1).fillna(0.0)
    gross = (weights_lag * asset_returns).sum(axis=1)
    turnover = weights_lag.diff().abs().sum(axis=1).fillna(0.0)
    tc = (float(tc_bps) / 10000.0) * turnover
    net = gross - tc
    equity = (1 + net).cumprod()
    return {
        "weights": weights,
        "weights_lag": weights_lag,
        "returns": net,
        "gross_returns": gross,
        "equity": equity,
        "turnover": turnover,
    }
