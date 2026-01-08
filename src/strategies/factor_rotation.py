from __future__ import annotations

"""
Factor Rotation (Blended, Monthly) strategy.

Assumptions:
- Monthly rebalancing at month-end using data available through that month.
- Weights are applied to next month's returns (no lookahead).
- Transaction costs are modeled as bps * turnover.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


ETF_UNIVERSE = ["QQQ", "SPY", "GLD", "TLT", "HYG"]


@dataclass
class RotationParams:
    momentum_w6: float = 0.6
    momentum_w12: float = 0.4
    trend_window: int = 10
    vol_window: int = 12
    trend_floor_cap: float = 0.05
    smooth_lambda: float = 0.35
    max_weight: float = 0.45
    tc_bps: float = 8.0
    vol_adjust: bool = True
    trend_filter: bool = True
    regime_tilt: bool = False


def monthly_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    return prices.resample("M").last()


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices_m = monthly_prices(prices)
    return prices_m.pct_change().replace([np.inf, -np.inf], np.nan)


def _rolling_total_return(returns: pd.Series, window: int) -> pd.Series:
    return (1 + returns).rolling(window).apply(lambda x: x.prod() - 1, raw=False)


def _softmax(scores: pd.Series) -> pd.Series:
    if scores.isna().all():
        return scores.fillna(0.0)
    vals = scores.fillna(scores.min())
    vals = vals - vals.max()
    exp = np.exp(vals)
    if exp.sum() == 0:
        return pd.Series(0.0, index=scores.index)
    return exp / exp.sum()


def _apply_trend_floor(raw: pd.Series, trend: pd.Series, floor: float) -> pd.Series:
    if trend.empty:
        return raw
    raw = raw.copy()
    off = trend <= 0
    if off.any():
        capped = raw.copy()
        capped.loc[off] = capped.loc[off].clip(upper=floor)
        keep = ~off
        remainder = 1.0 - capped.loc[off].sum()
        if remainder > 0 and keep.any():
            scaled = capped.loc[keep] / capped.loc[keep].sum() if capped.loc[keep].sum() > 0 else capped.loc[keep]
            capped.loc[keep] = scaled * remainder
        raw = capped
    return raw


def _apply_max_cap(weights: pd.Series, max_cap: float) -> pd.Series:
    weights = weights.clip(upper=max_cap)
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights


def compute_target_weights(
    prices: pd.DataFrame,
    params: RotationParams,
    regime_labels: pd.Series | None = None,
) -> pd.DataFrame:
    prices_m = monthly_prices(prices)
    rets_m = monthly_returns(prices)
    if prices_m.empty or rets_m.empty:
        return pd.DataFrame()

    mom6 = rets_m.apply(_rolling_total_return, window=6)
    mom12 = rets_m.apply(_rolling_total_return, window=12)
    scores = params.momentum_w6 * mom6 + params.momentum_w12 * mom12

    trend = pd.DataFrame(1.0, index=prices_m.index, columns=prices_m.columns)
    if params.trend_filter:
        ma10 = prices_m.rolling(params.trend_window).mean()
        trend = (prices_m > ma10).astype(float)

    vol = rets_m.rolling(params.vol_window).std()

    weights = []
    idx = []
    for dt in prices_m.index:
        score_row = scores.loc[dt]
        if score_row.dropna().empty:
            continue
        trend_row = trend.loc[dt]
        raw = _softmax(score_row)
        if params.trend_filter:
            raw = _apply_trend_floor(raw, trend_row, params.trend_floor_cap)
        if params.vol_adjust:
            vol_row = vol.loc[dt].replace(0, np.nan)
            if vol_row.dropna().empty:
                continue
            adj = raw / vol_row
            adj = adj.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            raw = adj / adj.sum() if adj.sum() > 0 else adj

        if params.regime_tilt and regime_labels is not None:
            regime_dt = regime_labels.reindex(prices_m.index).ffill().loc[dt]
            if regime_dt in ("Risk-Off / Credit Stress", "Rates Shock"):
                raw["TLT"] = min(params.max_weight, raw.get("TLT", 0) + 0.08)
                raw["GLD"] = min(params.max_weight, raw.get("GLD", 0) + 0.05)
                raw["QQQ"] = min(params.max_weight - 0.1, raw.get("QQQ", 0))
                raw = raw / raw.sum() if raw.sum() > 0 else raw

        raw = _apply_max_cap(raw, params.max_weight)
        weights.append(raw)
        idx.append(dt)

    weights_df = pd.DataFrame(weights, index=idx, columns=prices_m.columns).fillna(0.0)
    return weights_df


def apply_turnover_smoothing(weights: pd.DataFrame, smooth_lambda: float) -> pd.DataFrame:
    if weights.empty:
        return weights
    smooth = weights.copy()
    for i in range(1, len(weights)):
        smooth.iloc[i] = (1 - smooth_lambda) * smooth.iloc[i - 1] + smooth_lambda * weights.iloc[i]
        total = smooth.iloc[i].sum()
        if total > 0:
            smooth.iloc[i] = smooth.iloc[i] / total
    return smooth


def backtest_rotation(
    prices: pd.DataFrame,
    params: RotationParams,
    regime_labels: pd.Series | None = None,
) -> dict:
    prices = prices[ETF_UNIVERSE].dropna(how="all")
    if prices.empty:
        return {}

    weights = compute_target_weights(prices, params, regime_labels)
    if weights.empty:
        return {}

    weights = apply_turnover_smoothing(weights, params.smooth_lambda)
    rets_m = monthly_returns(prices).dropna(how="all")

    aligned_idx = weights.index.intersection(rets_m.index)
    weights = weights.reindex(aligned_idx).fillna(0.0)
    rets_m = rets_m.reindex(aligned_idx).fillna(0.0)

    weights_lag = weights.shift(1).fillna(0.0)
    gross = (weights_lag * rets_m).sum(axis=1)
    turnover = (weights_lag.diff().abs().sum(axis=1)).fillna(0.0)
    tc = (params.tc_bps / 10000.0) * turnover
    net = gross - tc

    equity = (1 + net).cumprod()
    return {
        "weights": weights,
        "weights_lag": weights_lag,
        "returns": net,
        "gross_returns": gross,
        "equity": equity,
        "turnover": turnover,
        "monthly_returns": rets_m,
    }
