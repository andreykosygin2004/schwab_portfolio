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
ALGORY_INITIAL_CASH = 100_000.0

_RF_WARNING = None


def load_holdings_timeseries(portfolio_id: str = "schwab") -> pd.DataFrame:
    if portfolio_id == "algory":
        return build_portfolio_timeseries(portfolio_id=portfolio_id)
    if not TRANSACTIONS_CSV.exists():
        return _build_hypothetical_timeseries()
    if not HOLDINGS_TS.exists():
        return pd.DataFrame()
    return pd.read_csv(HOLDINGS_TS, parse_dates=["Date"], index_col="Date").sort_index()


def load_transactions(portfolio_id: str = "schwab") -> pd.DataFrame:
    if portfolio_id == "algory" or not TRANSACTIONS_CSV.exists():
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
    return pd.read_csv(TRANSACTIONS_CSV)


def risk_free_warning() -> str | None:
    return _RF_WARNING


def _set_rf_warning(message: str | None) -> None:
    global _RF_WARNING
    _RF_WARNING = message


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
        _set_rf_warning(None)
        return rf_return

    from analytics_macro import load_ticker_prices
    proxies = ["BIL", "SHV"]
    proxy_prices = load_ticker_prices(proxies, start=index.min(), end=index.max())
    if not proxy_prices.empty:
        proxy = proxy_prices.iloc[:, 0]
        rf_return = proxy.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        rf_return = rf_return.reindex(index).ffill().fillna(0.0)
        _set_rf_warning(None)
        return rf_return

    _set_rf_warning("Risk-free unavailable; cash assumed 0%.")
    warnings.warn("[WARN] Risk-free unavailable; cash assumed 0%.")
    return pd.Series(0.0, index=index)


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
    start = portfolio["entry_date"].min() - pd.Timedelta(days=7)
    end = pd.Timestamp.today().normalize()
    from analytics_macro import load_ticker_prices
    prices = load_ticker_prices(tickers, start=start, end=end)
    if prices.empty:
        return pd.DataFrame()
    if prices.empty:
        return pd.DataFrame()
    prices = prices.sort_index()
    prices = prices.ffill()
    idx = prices.index
    holdings = {}
    cash = pd.Series(ALGORY_INITIAL_CASH, index=idx)
    for _, row in portfolio.iterrows():
        symbol = row["symbol"]
        entry_date = row["entry_date"]
        shares = float(row["shares"])
        if symbol not in prices.columns:
            continue
        price_series = prices[symbol].dropna()
        trade_dates = price_series.index[price_series.index >= entry_date]
        if trade_dates.empty:
            continue
        trade_date = trade_dates[0]
        if symbol not in holdings:
            holdings[symbol] = pd.Series(0.0, index=idx)
        holdings[symbol].loc[idx >= trade_date] = shares
        hist_price = prices.loc[trade_date, symbol]
        if pd.notna(hist_price):
            cash.loc[trade_date:] -= shares * float(hist_price)
        entry_price = float(row["entry_price"]) if pd.notna(row.get("entry_price")) else np.nan
        if pd.notna(entry_price) and pd.notna(hist_price) and hist_price != 0:
            diff = abs(entry_price - hist_price) / hist_price
            if diff > 0.2:
                warnings.warn(
                    f"[WARN] Algory entry price for {symbol} on {trade_date.date()} "
                    f"differs from adjusted close {hist_price:.2f} by {diff:.0%}."
                )
    shares_df = pd.DataFrame(holdings).reindex(idx).fillna(0.0)
    mv_df = shares_df * prices.reindex(shares_df.index).fillna(0.0)
    out = pd.DataFrame(index=idx)
    for col in mv_df.columns:
        out[f"MV_{col}"] = mv_df[col]
    out["portfolio_value"] = mv_df.sum(axis=1)
    out["cash_balance"] = cash
    out["cash_balance_clean"] = cash
    out["total_value"] = out["portfolio_value"] + cash
    out["total_value_rf"] = out["portfolio_value"] + cash
    out["total_value_clean"] = out["portfolio_value"] + cash
    out["total_value_clean_rf"] = out["portfolio_value"] + cash
    return out


def _extend_holdings_to_present(holdings_ts: pd.DataFrame) -> pd.DataFrame:
    if holdings_ts.empty:
        return holdings_ts

    last_date = holdings_ts.index.max()
    today = pd.Timestamp.today().normalize()
    if last_date >= today:
        return holdings_ts

    mv_cols = [c for c in holdings_ts.columns if c.startswith("MV_")]
    if not mv_cols:
        return holdings_ts

    tickers = [c.replace("MV_", "") for c in mv_cols]
    from analytics_macro import load_ticker_prices
    prices = load_ticker_prices(tickers, start=last_date - pd.Timedelta(days=10), end=today)
    if prices.empty:
        return holdings_ts
    prices = prices.sort_index().ffill()

    last_prices = {}
    for t in tickers:
        series = prices[t].dropna()
        last_prices[t] = series.loc[:last_date].iloc[-1] if not series.loc[:last_date].empty else np.nan

    shares = {}
    last_row = holdings_ts.loc[last_date, mv_cols]
    for col in mv_cols:
        ticker = col.replace("MV_", "")
        price = last_prices.get(ticker, np.nan)
        mv = float(last_row.get(col, 0.0))
        shares[ticker] = mv / price if pd.notna(price) and price != 0 else 0.0

    future_idx = prices.index[prices.index > last_date]
    if future_idx.empty:
        return holdings_ts

    future_prices = prices.loc[future_idx, tickers]
    mv_future = pd.DataFrame({f"MV_{t}": future_prices[t] * shares[t] for t in tickers}, index=future_idx)
    future = pd.DataFrame(index=future_idx)
    for col in mv_cols:
        future[col] = mv_future[col]
    future["portfolio_value"] = mv_future.sum(axis=1)
    for col in ["cash_balance", "cash_balance_clean", "cumulative_deposits"]:
        if col in holdings_ts.columns:
            future[col] = float(holdings_ts.loc[last_date, col])
    if "total_value" in holdings_ts.columns:
        cash_bal = future.get("cash_balance", 0.0)
        future["total_value"] = future["portfolio_value"] + cash_bal
    if "total_value_clean" in holdings_ts.columns:
        cash_clean = future.get("cash_balance_clean", future.get("cash_balance", 0.0))
        future["total_value_clean"] = future["portfolio_value"] + cash_clean

    return pd.concat([holdings_ts, future], axis=0).sort_index()


def build_portfolio_timeseries(freq: str = "Daily", portfolio_id: str = "schwab") -> pd.DataFrame:
    if portfolio_id == "algory":
        holdings_ts = _build_hypothetical_timeseries()
    else:
        holdings_ts = load_holdings_timeseries()
    if holdings_ts.empty:
        return holdings_ts
    if portfolio_id != "algory":
        holdings_ts = _extend_holdings_to_present(holdings_ts)

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


def get_portfolio_date_bounds(portfolio_id: str = "schwab") -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    df = build_portfolio_timeseries(portfolio_id=portfolio_id)
    if df.empty:
        return None, None
    return df.index.min(), df.index.max()
