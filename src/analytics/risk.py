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
    episodes = []
    if (dd < 0).sum() == 0:
        return episodes

    peak_pos = 0
    peak_date = dd.index[0]
    in_episode = False
    trough_pos = None
    trough_date = None
    trough_val = 0.0
    last_pos = len(dd.index) - 1

    for pos, (date, value) in enumerate(dd.items()):
        if np.isclose(value, 0.0, atol=1e-12):
            if in_episode:
                episodes.append(
                    _episode_record(peak_date, peak_pos, trough_date, trough_pos, date, pos, dd, last_pos)
                )
                in_episode = False
            peak_pos = pos
            peak_date = date
            trough_pos = None
            trough_date = None
            trough_val = 0.0
            continue

        if value < 0:
            if not in_episode:
                in_episode = True
                trough_pos = pos
                trough_date = date
                trough_val = value
            elif value < trough_val:
                trough_val = value
                trough_pos = pos
                trough_date = date

    if in_episode and trough_date is not None and trough_pos is not None:
        episodes.append(
            _episode_record(peak_date, peak_pos, trough_date, trough_pos, None, None, dd, last_pos)
        )

    episodes = sorted(episodes, key=lambda x: x["depth"])
    return episodes


def _episode_record(
    peak_date: pd.Timestamp,
    peak_pos: int,
    trough_date: pd.Timestamp,
    trough_pos: int,
    recovery_date: pd.Timestamp | None,
    recovery_pos: int | None,
    dd: pd.Series,
    last_pos: int,
) -> dict:
    depth = float(dd.loc[trough_date])
    duration_to_trough = trough_pos - peak_pos
    if recovery_date is not None and recovery_pos is not None:
        recovery_time = recovery_pos - trough_pos
        total_duration = recovery_pos - peak_pos
    else:
        recovery_time = None
        total_duration = last_pos - peak_pos

    return {
        "peak": peak_date,
        "trough": trough_date,
        "recovery": recovery_date,
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
