from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(prices: pd.Series | pd.DataFrame, freq: str) -> pd.Series | pd.DataFrame:
    prices = prices.sort_index()
    if freq == "Weekly":
        prices = prices.resample("W-FRI").last()
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return returns


def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        return np.nan
    return float(returns.std() * np.sqrt(periods_per_year))


def drawdown_series(prices: pd.Series) -> pd.Series:
    if prices.empty:
        return pd.Series(dtype=float)
    roll_max = prices.cummax()
    return (prices / roll_max) - 1.0


def max_drawdown(prices: pd.Series) -> float:
    dd = drawdown_series(prices)
    return float(dd.min()) if not dd.empty else np.nan


def drawdown_episodes(prices: pd.Series) -> list[dict]:
    if prices.empty:
        return []
    dd = drawdown_series(prices)
    in_dd = dd < 0
    episodes = []
    if not in_dd.any():
        return episodes

    start = None
    trough = None
    trough_val = 0.0

    for date, is_dd in in_dd.items():
        if is_dd and start is None:
            start = date
            trough = date
            trough_val = dd.loc[date]
        elif is_dd:
            if dd.loc[date] < trough_val:
                trough_val = dd.loc[date]
                trough = date
        elif not is_dd and start is not None:
            recovery = date
            episodes.append(_episode_record(start, trough, recovery, dd))
            start = None
            trough = None
            trough_val = 0.0

    if start is not None:
        episodes.append(_episode_record(start, trough, None, dd))

    episodes = sorted(episodes, key=lambda x: x["depth"], reverse=True)
    return episodes


def _episode_record(start: pd.Timestamp, trough: pd.Timestamp, recovery: pd.Timestamp | None, dd: pd.Series) -> dict:
    depth = float(dd.loc[trough])
    duration_to_trough = (trough - start).days
    if recovery is not None:
        recovery_time = (recovery - trough).days
        total_duration = (recovery - start).days
    else:
        recovery_time = None
        total_duration = (dd.index[-1] - start).days

    return {
        "start": start,
        "trough": trough,
        "recovery": recovery,
        "depth": depth,
        "duration_to_trough": duration_to_trough,
        "recovery_time": recovery_time,
        "total_duration": total_duration,
    }


def var_cvar(returns: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    if returns.empty:
        return np.nan, np.nan
    q = returns.quantile(alpha)
    cvar = returns[returns <= q].mean()
    return float(q), float(cvar)


def rolling_var(returns: pd.Series, window: int, alpha: float = 0.05) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    return returns.rolling(window).quantile(alpha)
