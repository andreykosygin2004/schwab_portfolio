"""Analytics helper modules."""

from .risk import (
    annualized_vol,
    compute_returns,
    drawdown_episodes,
    drawdown_series,
    max_drawdown,
    rolling_var,
    var_cvar,
)

import importlib.util
from pathlib import Path


_ANALYTICS_PATH = Path(__file__).resolve().parents[1] / "data_creation.py"
_spec = importlib.util.spec_from_file_location("analytics_module", _ANALYTICS_PATH)
_analytics = importlib.util.module_from_spec(_spec) if _spec else None
if _spec and _spec.loader:
    _spec.loader.exec_module(_analytics)
    compute_performance_metrics = _analytics.compute_performance_metrics
    compute_clean_cash_balance = _analytics.compute_clean_cash_balance
else:
    compute_performance_metrics = None
    compute_clean_cash_balance = None

__all__ = [
    "annualized_vol",
    "compute_returns",
    "drawdown_episodes",
    "drawdown_series",
    "max_drawdown",
    "rolling_var",
    "var_cvar",
    "compute_performance_metrics",
    "compute_clean_cash_balance",
]
