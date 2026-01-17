from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
HOLDINGS_TS = DATA_DIR / "holdings_timeseries.csv"
TREASURY_CSV = DATA_DIR / "treasury.csv"
HYPOTHETICAL_PORTFOLIO = DATA_DIR / "hypothetical_portfolio.csv"
TRANSACTIONS_CSV = DATA_DIR / "schwab_transactions.csv"

_RF_WARNING = None
_RF_SOURCE = "unknown"


def load_holdings_timeseries(portfolio_id: str = "schwab") -> pd.DataFrame:
    if portfolio_id == "algory":
        return build_portfolio_timeseries(portfolio_id=portfolio_id)
    return pd.read_csv(HOLDINGS_TS, parse_dates=["Date"], index_col="Date").sort_index()


def load_transactions(portfolio_id: str = "schwab") -> pd.DataFrame:
    if portfolio_id == "algory":
        portfolio = _load_hypothetical_portfolio()
        if portfolio.empty:
            return pd.DataFrame()
        rows = []
        for _, row in portfolio.iterrows():
            shares = float(row["shares"])
            price = float(row["entry_price"]) if pd.notna(row.get("entry_price")) else np.nan
            amount = -shares * price if pd.notna(price) else np.nan
            rows.append({
                "Date": row["entry_date"].date().isoformat(),
                "Action": "Buy",
                "Symbol": row["symbol"],
                "Quantity": shares,
                "Price": price,
                "Amount": amount,
            })
        return pd.DataFrame(rows)
    if not TRANSACTIONS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(TRANSACTIONS_CSV)


def risk_free_warning() -> str | None:
    return _RF_WARNING


def risk_free_source() -> str:
    return _RF_SOURCE


def _set_rf_warning(message: str | None) -> None:
    global _RF_WARNING
    _RF_WARNING = message


def _set_rf_source(source: str) -> None:
    global _RF_SOURCE
    _RF_SOURCE = source


@lru_cache(maxsize=4)
def _load_treasury_yield() -> pd.Series | None:
    if not TREASURY_CSV.exists():
        return None
    df = pd.read_csv(TREASURY_CSV)
    df.columns = [c.strip() for c in df.columns]
    date_col = next((c for c in df.columns if c.strip().lower() == "date"), None)
    if date_col is None:
        return None
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    if df.empty:
        return None
    if "DGS10" in df.columns:
        series = pd.to_numeric(df["DGS10"], errors="coerce")
    else:
        numeric_cols = [c for c in df.columns if c != date_col]
        if not numeric_cols:
            return None
        series = pd.to_numeric(df[numeric_cols[0]], errors="coerce")
    series = series.dropna()
    if series.empty:
        return None
    return series


def load_risk_free_returns(index: pd.DatetimeIndex, freq: str) -> pd.Series:
    if index.empty:
        return pd.Series(dtype=float)

    periods = 252 if freq == "Daily" else 52
    treasury_yield = _load_treasury_yield()
    if treasury_yield is not None:
        rf_annual = treasury_yield / 100.0
        rf_return = (1.0 + rf_annual).pow(1.0 / periods) - 1.0
        rf_return = rf_return.reindex(index).ffill().fillna(0.0)
        _set_rf_source("treasury")
        _set_rf_warning(None)
        return rf_return

    from analytics_macro import load_ticker_prices
    proxies = ["BIL", "SHV"]
    proxy_prices = load_ticker_prices(proxies, start=index.min(), end=index.max())
    if not proxy_prices.empty:
        proxy = proxy_prices.iloc[:, 0]
        rf_return = proxy.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        rf_return = rf_return.reindex(index).ffill().fillna(0.0)
        _set_rf_source(f"proxy:{proxy.name}")
        _set_rf_warning(None)
        return rf_return

    _set_rf_source("fallback:zero")
    _set_rf_warning("Risk-free unavailable; cash assumed 0%.")
    warnings.warn("[WARN] Risk-free unavailable; cash assumed 0%.")
    return pd.Series(0.0, index=index)


def risk_free_info(index: pd.DatetimeIndex, freq: str) -> dict:
    rf = load_risk_free_returns(index, freq)
    non_zero = bool((rf != 0).any())
    last = float(rf.dropna().iloc[-1]) if not rf.dropna().empty else 0.0
    return {"source": _RF_SOURCE, "non_zero": non_zero, "last_return": last}


def accrue_cash_balance(cash_balance: pd.Series, rf_returns: pd.Series) -> pd.Series:
    if cash_balance.empty:
        return pd.Series(dtype=float)
    cash_balance = cash_balance.reindex(rf_returns.index).ffill().fillna(0.0)
    flows = cash_balance.diff().fillna(cash_balance.iloc[0])
    cash_value = []
    prev = 0.0
    for date, flow in flows.items():
        rf = rf_returns.loc[date] if date in rf_returns.index else 0.0
        if prev > 0:
            prev = prev * (1.0 + rf)
        prev += float(flow)
        cash_value.append(prev)
    return pd.Series(cash_value, index=flows.index)


def _load_hypothetical_portfolio() -> pd.DataFrame:
    if not HYPOTHETICAL_PORTFOLIO.exists():
        return pd.DataFrame()
    df = pd.read_csv(HYPOTHETICAL_PORTFOLIO)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"share name": "symbol"})
    if "symbol" not in df.columns or "entry_date" not in df.columns:
        return pd.DataFrame()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["shares"] = pd.to_numeric(df.get("shares"), errors="coerce")
    df["entry_price"] = pd.to_numeric(df.get("entry_price"), errors="coerce")
    return df.dropna(subset=["symbol", "entry_date", "shares"])


def _build_hypothetical_timeseries() -> pd.DataFrame:
    portfolio = _load_hypothetical_portfolio()
    if portfolio.empty:
        return pd.DataFrame()
    tickers = portfolio["symbol"].unique().tolist()
    start = portfolio["entry_date"].min()
    end = pd.Timestamp.today().normalize()
    from analytics_macro import load_ticker_prices
    prices = load_ticker_prices(tickers, start=start, end=end)
    if prices.empty:
        return pd.DataFrame()
    idx = prices.index
    holdings = {}
    for _, row in portfolio.iterrows():
        symbol = row["symbol"]
        entry_date = row["entry_date"]
        shares = float(row["shares"])
        if symbol not in prices.columns:
            continue
        if symbol not in holdings:
            holdings[symbol] = pd.Series(0.0, index=idx)
        holdings[symbol].loc[idx >= entry_date] = shares
    shares_df = pd.DataFrame(holdings).reindex(idx).fillna(0.0)
    mv_df = shares_df * prices.reindex(shares_df.index).fillna(0.0)
    out = pd.DataFrame(index=idx)
    for col in mv_df.columns:
        out[f"MV_{col}"] = mv_df[col]
    out["portfolio_value"] = mv_df.sum(axis=1)
    cash_value = 5345.0
    out["cash_balance"] = cash_value
    out["cash_balance_clean"] = cash_value
    out["total_value"] = out["portfolio_value"] + cash_value
    out["total_value_rf"] = out["portfolio_value"] + cash_value
    out["total_value_clean"] = out["portfolio_value"] + cash_value
    out["total_value_clean_rf"] = out["portfolio_value"] + cash_value
    return out


def build_portfolio_timeseries(freq: str = "Daily", portfolio_id: str = "schwab") -> pd.DataFrame:
    if portfolio_id == "algory":
        return _build_hypothetical_timeseries()
    holdings_ts = load_holdings_timeseries()
    if holdings_ts.empty:
        return holdings_ts

    index = holdings_ts.index
    rf_returns = load_risk_free_returns(index, freq="Daily")
    cash_balance = holdings_ts["cash_balance"] if "cash_balance" in holdings_ts.columns else pd.Series(0.0, index=index)
    cash_clean = holdings_ts.get("cash_balance_clean", cash_balance)

    cash_value_rf = accrue_cash_balance(cash_balance, rf_returns)
    cash_clean_rf = accrue_cash_balance(cash_clean, rf_returns)

    portfolio_value = holdings_ts["portfolio_value"] if "portfolio_value" in holdings_ts.columns else None
    if portfolio_value is None:
        return holdings_ts

    holdings_ts = holdings_ts.copy()
    holdings_ts["cash_value_rf"] = cash_value_rf
    holdings_ts["cash_value_clean_rf"] = cash_clean_rf
    holdings_ts["total_value_rf"] = portfolio_value + cash_value_rf
    holdings_ts["total_value_clean_rf"] = portfolio_value + cash_clean_rf
    return holdings_ts


def load_portfolio_series(series_name: str = "total_value_clean_rf", portfolio_id: str = "schwab") -> pd.Series:
    df = build_portfolio_timeseries(portfolio_id=portfolio_id)
    if series_name in df.columns:
        return df[series_name].astype(float)
    if "total_value_clean" in df.columns:
        return df["total_value_clean"].astype(float)
    if "total_value" in df.columns:
        return df["total_value"].astype(float)
    raise ValueError("No usable portfolio series found in holdings timeseries.")
