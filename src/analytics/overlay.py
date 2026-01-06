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


def rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    y, x = align_series(y, x)
    if y.empty:
        return pd.Series(dtype=float)
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / var


def compute_overlay_weights(
    labels: pd.Series,
    beta_series: pd.Series,
    config: dict,
) -> pd.DataFrame:
    if labels.empty or beta_series.empty:
        return pd.DataFrame()
    labels = labels.reindex(beta_series.index).ffill()

    rows = []
    for dt, label in labels.items():
        beta = beta_series.loc[dt]
        target_mult = config["target_beta_mult"].get(label, 1.0)
        target_beta = beta * target_mult
        hedge_beta_needed = beta - target_beta
        w_qqq = min(max(hedge_beta_needed, -config["max_hedge"]), config["max_hedge"])
        w_tlt = config["tlt_tilt"].get(label, 0.0)
        w_gld = config["gld_tilt"].get(label, 0.0)

        gross = abs(w_qqq) + abs(w_tlt) + abs(w_gld)
        gross_cap = config["gross_cap"]
        if not config["allow_leverage"]:
            gross_cap = min(gross_cap, 1.0)
        if gross > gross_cap and gross > 0:
            scale = gross_cap / gross
            w_qqq *= scale
            w_tlt *= scale
            w_gld *= scale

        rows.append({"w_qqq": w_qqq, "w_tlt": w_tlt, "w_gld": w_gld})
    return pd.DataFrame(rows, index=beta_series.index)


def apply_overlay(
    port_ret: pd.Series,
    qqq_ret: pd.Series,
    tlt_ret: pd.Series | None,
    gld_ret: pd.Series | None,
    weights: pd.DataFrame,
) -> pd.Series:
    aligned = align_series(port_ret, qqq_ret)
    port_ret, qqq_ret = aligned
    if port_ret.empty:
        return pd.Series(dtype=float)
    weights = weights.reindex(port_ret.index).fillna(0.0)
    overlay = port_ret - weights["w_qqq"] * qqq_ret
    if tlt_ret is not None and not tlt_ret.empty:
        tlt_ret = tlt_ret.reindex(port_ret.index).fillna(0.0)
        overlay = overlay + weights["w_tlt"] * tlt_ret
    if gld_ret is not None and not gld_ret.empty:
        gld_ret = gld_ret.reindex(port_ret.index).fillna(0.0)
        overlay = overlay + weights["w_gld"] * gld_ret
    return overlay


def overlay_summary_stats(
    base_ret: pd.Series,
    overlay_ret: pd.Series,
    weights: pd.DataFrame,
    qqq_ret: pd.Series,
    periods: int,
    beta_window: int,
) -> dict:
    base_ret, overlay_ret, qqq_ret = align_series(base_ret, overlay_ret, qqq_ret)
    if base_ret.empty:
        return {}
    avg_hedge = float(weights["w_qqq"].abs().mean()) if not weights.empty else 0.0
    max_hedge = float(weights["w_qqq"].abs().max()) if not weights.empty else 0.0
    pct_hedged = float((weights["w_qqq"] > 0).mean()) if not weights.empty else 0.0
    turnover = float(weights.diff().abs().sum(axis=1).mean()) if not weights.empty else 0.0
    rolling_beta_overlay = rolling_beta(overlay_ret, qqq_ret, beta_window)
    avg_beta = float(rolling_beta_overlay.dropna().mean()) if not rolling_beta_overlay.empty else np.nan
    return {
        "avg_hedge": avg_hedge,
        "max_hedge": max_hedge,
        "pct_hedged": pct_hedged,
        "avg_beta": avg_beta,
        "turnover": turnover,
    }
