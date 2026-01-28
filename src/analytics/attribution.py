from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.common import time_varying_weights


def compute_contributions(weights: pd.Series, returns: pd.Series) -> pd.Series:
    weights = weights.reindex(returns.index).fillna(0.0)
    return weights * returns


def top_contributors(contrib: pd.Series, n: int) -> pd.Series:
    if contrib.empty:
        return contrib
    return contrib.reindex(contrib.abs().sort_values(ascending=False).head(n).index)


def factor_period_contributions(betas: pd.Series, factor_returns: pd.DataFrame) -> pd.DataFrame:
    if betas.empty or factor_returns.empty:
        return pd.DataFrame()
    aligned = factor_returns.reindex(columns=betas.index).dropna(how="all")
    return aligned.multiply(betas, axis=1)


def build_pm_memo(metrics: dict) -> list[str]:
    memo = []
    if metrics.get("top_contrib"):
        memo.append(f"Top contributor: {metrics['top_contrib']}.")
    if metrics.get("top_detractor"):
        memo.append(f"Largest detractor: {metrics['top_detractor']}.")
    if metrics.get("top_factor"):
        memo.append(f"Dominant factor: {metrics['top_factor']}.")
    if metrics.get("residual"):
        memo.append(f"Residual return over window: {metrics['residual']}.")
    if metrics.get("regime_note"):
        memo.append(metrics["regime_note"])
    return memo


def time_series_attribution(
    mv_df: pd.DataFrame,
    price_df: pd.DataFrame,
    total_value: pd.Series,
) -> pd.DataFrame:
    if mv_df.empty or price_df.empty or total_value.empty:
        return pd.DataFrame()

    weights_lag = time_varying_weights(mv_df, freq="Daily", total_value=total_value)

    prices = price_df.reindex(weights_lag.index).ffill().dropna(how="all")
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    aligned = weights_lag.index.intersection(returns.index)
    weights_lag = weights_lag.reindex(aligned).fillna(0.0)
    returns = returns.reindex(aligned)

    contrib = weights_lag * returns
    total_contrib = contrib.sum(axis=0)
    avg_weight = weights_lag.mean(axis=0)
    total_return = (1 + returns).prod() - 1

    out = pd.DataFrame({
        "avg_weight": avg_weight,
        "total_return": total_return,
        "contribution": total_contrib,
    })
    out["pct_total"] = out["contribution"] / out["contribution"].sum() if out["contribution"].sum() != 0 else 0.0
    return out.sort_values("contribution", ascending=False)


def compute_trade_based_returns(
    transactions: pd.DataFrame,
    price_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    if transactions.empty:
        return pd.Series(dtype=float)

    tx = transactions.copy()
    tx["Date"] = pd.to_datetime(tx["Date"], errors="coerce")
    tx = tx.dropna(subset=["Date"])

    def _to_num(val):
        if pd.isna(val):
            return np.nan
        return pd.to_numeric(str(val).replace("$", "").replace(",", ""), errors="coerce")

    tx["Quantity"] = tx["Quantity"].apply(_to_num)
    tx["Price"] = tx["Price"].apply(_to_num)
    tx = tx[tx["Symbol"].notna()]

    buys = tx[tx["Action"].str.lower() == "buy"].copy()
    sells = tx[tx["Action"].str.lower() == "sell"].copy()

    out = {}
    for symbol in tx["Symbol"].dropna().unique():
        sym_buys = buys[buys["Symbol"] == symbol]
        sym_sells = sells[sells["Symbol"] == symbol]
        if sym_buys.empty:
            continue

        last_sell_date = sym_sells[(sym_sells["Date"] >= start) & (sym_sells["Date"] <= end)]["Date"].max()
        cutoff = last_sell_date if pd.notna(last_sell_date) else end

        buy_slice = sym_buys[sym_buys["Date"] <= cutoff]
        if buy_slice.empty:
            continue
        buy_qty = buy_slice["Quantity"].abs().sum()
        buy_px = (buy_slice["Quantity"].abs() * buy_slice["Price"]).sum() / buy_qty if buy_qty > 0 else np.nan

        sell_slice = sym_sells[(sym_sells["Date"] >= start) & (sym_sells["Date"] <= end)]
        if not sell_slice.empty:
            sell_qty = sell_slice["Quantity"].abs().sum()
            sell_px = (sell_slice["Quantity"].abs() * sell_slice["Price"]).sum() / sell_qty if sell_qty > 0 else np.nan
        else:
            if symbol in price_df.columns:
                series = price_df[symbol].dropna()
                series = series.loc[:end]
                sell_px = float(series.iloc[-1]) if not series.empty else np.nan
            else:
                sell_px = np.nan

        if pd.notna(buy_px) and pd.notna(sell_px) and buy_px > 0:
            out[symbol] = (sell_px / buy_px) - 1

    return pd.Series(out)
