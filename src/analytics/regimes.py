from __future__ import annotations

from functools import lru_cache
import numpy as np
import pandas as pd

from analytics.constants import ANALYSIS_END, ANALYSIS_START
from analytics.risk import compute_returns, drawdown_series
from analytics_macro import load_ticker_prices


PROXY_TICKERS = ["SPY", "QQQ", "HYG", "TLT", "USO", "UUP", "GLD", "TIP"]


def analysis_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    return ANALYSIS_START, ANALYSIS_END


@lru_cache(maxsize=4)
def load_proxy_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return load_ticker_prices(PROXY_TICKERS, start=start, end=end)


def clear_proxy_cache() -> None:
    load_proxy_prices.cache_clear()


def compute_regime_features(proxy_prices: pd.DataFrame, freq: str) -> pd.DataFrame:
    if proxy_prices.empty:
        return pd.DataFrame()

    prices = proxy_prices.copy()
    if freq == "Weekly":
        prices = prices.resample("W-FRI").last()

    returns = compute_returns(prices, freq="Daily")

    features = pd.DataFrame(index=returns.index)

    if "SPY" in prices.columns:
        mom_window = 63 if freq == "Daily" else 12
        vol_window = 20 if freq == "Daily" else 12
        features["spy_mom_63"] = prices["SPY"].pct_change(mom_window)
        features["spy_vol_20"] = returns["SPY"].rolling(vol_window).std()
    if "QQQ" in prices.columns:
        mom_window = 63 if freq == "Daily" else 12
        features["qqq_mom_63"] = prices["QQQ"].pct_change(mom_window)
    if "HYG" in prices.columns:
        mom_window = 21 if freq == "Daily" else 8
        features["hyg_mom_21"] = prices["HYG"].pct_change(mom_window)
        features["hyg_dd"] = drawdown_series(prices["HYG"])
    if "TLT" in prices.columns:
        mom_window = 21 if freq == "Daily" else 8
        features["tlt_mom_21"] = prices["TLT"].pct_change(mom_window)
        features["tlt_dd"] = drawdown_series(prices["TLT"])
    if "UUP" in prices.columns:
        mom_window = 21 if freq == "Daily" else 8
        features["uup_mom_21"] = prices["UUP"].pct_change(mom_window)
    if "USO" in prices.columns:
        mom_window = 21 if freq == "Daily" else 8
        features["uso_mom_21"] = prices["USO"].pct_change(mom_window)
    if "TIP" in prices.columns:
        mom_window = 21 if freq == "Daily" else 8
        features["tip_mom_21"] = prices["TIP"].pct_change(mom_window)

    features = features.dropna(how="all")
    return features


def resample_prices(prices: pd.Series | pd.DataFrame, freq: str) -> pd.Series | pd.DataFrame:
    if freq == "Weekly":
        return prices.resample("W-FRI").last()
    return prices


def returns_from_prices(prices: pd.Series | pd.DataFrame, freq: str) -> pd.Series | pd.DataFrame:
    prices = resample_prices(prices, freq)
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return returns


def get_thresholds(preset: str) -> dict[str, float]:
    if preset == "Conservative":
        return {
            "mom_pos": 0.03,
            "vol_high": 0.02,
            "credit_mom_neg": -0.03,
            "credit_dd": -0.08,
            "tlt_dd": -0.08,
            "uso_mom": 0.08,
            "tip_weak": -0.02,
        }
    if preset == "Aggressive":
        return {
            "mom_pos": 0.01,
            "vol_high": 0.018,
            "credit_mom_neg": -0.015,
            "credit_dd": -0.05,
            "tlt_dd": -0.05,
            "uso_mom": 0.05,
            "tip_weak": -0.01,
        }
    return {
        "mom_pos": 0.02,
        "vol_high": 0.019,
        "credit_mom_neg": -0.02,
        "credit_dd": -0.06,
        "tlt_dd": -0.06,
        "uso_mom": 0.06,
        "tip_weak": -0.015,
    }


def label_regimes(features: pd.DataFrame, preset: str) -> pd.Series:
    if features.empty:
        return pd.Series(dtype=str)
    th = get_thresholds(preset)

    labels = []
    for _, row in features.iterrows():
        spy_mom = row.get("spy_mom_63", np.nan)
        qqq_mom = row.get("qqq_mom_63", np.nan)
        vol = row.get("spy_vol_20", np.nan)
        hyg_mom = row.get("hyg_mom_21", np.nan)
        hyg_dd = row.get("hyg_dd", np.nan)
        tlt_dd = row.get("tlt_dd", np.nan)
        uso_mom = row.get("uso_mom_21", np.nan)
        tip_mom = row.get("tip_mom_21", np.nan)

        credit_stress = (
            (not np.isnan(hyg_mom) and hyg_mom <= th["credit_mom_neg"]) or
            (not np.isnan(hyg_dd) and hyg_dd <= th["credit_dd"]) or
            (not np.isnan(vol) and vol >= th["vol_high"])
        )
        rates_shock = not np.isnan(tlt_dd) and tlt_dd <= th["tlt_dd"]
        inflation_shock = (
            (not np.isnan(uso_mom) and uso_mom >= th["uso_mom"]) and
            (not np.isnan(tip_mom) and tip_mom <= th["tip_weak"])
        )
        risk_on = (
            ((not np.isnan(spy_mom) and spy_mom >= th["mom_pos"]) or
             (not np.isnan(qqq_mom) and qqq_mom >= th["mom_pos"])) and
            (not np.isnan(vol) and vol < th["vol_high"]) and
            not credit_stress
        )

        if credit_stress:
            labels.append("Risk-Off / Credit Stress")
        elif rates_shock:
            labels.append("Rates Shock")
        elif inflation_shock:
            labels.append("Inflation Shock")
        elif risk_on:
            labels.append("Risk-On")
        else:
            labels.append("Neutral / Transition")

    return pd.Series(labels, index=features.index)


def simulate_overlay(
    port_returns: pd.Series,
    regime_labels: pd.Series,
    exposure_map: dict[str, float],
    hedge_weights: dict[str, float] | None = None,
    hedge_returns: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series]:
    aligned_idx = port_returns.index.intersection(regime_labels.index)
    port_returns = port_returns.reindex(aligned_idx)
    labels = regime_labels.reindex(aligned_idx)
    exposure = labels.map(lambda x: exposure_map.get(x, 1.0)).fillna(1.0)
    overlay = port_returns * exposure

    if hedge_weights and hedge_returns is not None and not hedge_returns.empty:
        hedge_returns = hedge_returns.reindex(aligned_idx).fillna(0.0)
        hedge_component = pd.Series(0.0, index=aligned_idx)
        for hedge, weight in hedge_weights.items():
            if hedge in hedge_returns.columns:
                hedge_component += hedge_returns[hedge] * weight
        overlay = overlay + hedge_component

    return overlay, exposure


def transition_matrix(labels: pd.Series) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()
    states = labels.dropna().unique().tolist()
    matrix = pd.DataFrame(0.0, index=states, columns=states)
    for prev, nxt in zip(labels.iloc[:-1], labels.iloc[1:]):
        if prev in matrix.index and nxt in matrix.columns:
            matrix.loc[prev, nxt] += 1
    matrix = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0)
    return matrix.fillna(0.0)
