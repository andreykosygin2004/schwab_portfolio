from __future__ import annotations

import warnings
import numpy as np
import pandas as pd


def _to_number(value) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).replace("$", "").replace(",", "").strip()
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    return pd.to_numeric(s, errors="coerce")


def infer_shares_from_amount(amount: float, price: float) -> tuple[int, float, bool]:
    if price <= 0 or np.isnan(price) or np.isnan(amount):
        return 0, np.nan, False
    shares_raw = abs(amount) / price
    shares = max(1, int(round(shares_raw)))
    implied_fill = abs(amount) / shares if shares > 0 else np.nan
    confidence = abs(shares_raw - round(shares_raw)) <= 0.15
    if np.isnan(implied_fill) or abs(implied_fill - price) / price > 0.03:
        confidence = False
    return shares, implied_fill, confidence


def _price_on_or_before(prices: pd.Series, date: pd.Timestamp) -> tuple[float | None, str]:
    if prices.empty:
        return None, "missing"
    if date in prices.index and pd.notna(prices.loc[date]):
        return float(prices.loc[date]), "exact"
    prior = prices.loc[:date].dropna()
    if prior.empty:
        return None, "missing"
    return float(prior.iloc[-1]), "prior"


def build_starting_positions_from_mv(
    mv_df: pd.DataFrame,
    prices: pd.DataFrame,
    start: pd.Timestamp,
) -> pd.Series:
    if mv_df.empty or prices.empty:
        return pd.Series(dtype=float)
    mv_df = mv_df.copy()
    mv_df.columns = [c.replace("MV_", "") for c in mv_df.columns]
    mv_df = mv_df.sort_index()
    mv_slice = mv_df.loc[:start]
    if mv_slice.empty:
        return pd.Series(dtype=float)
    mv_row = mv_slice.iloc[-1].fillna(0.0)
    shares = {}
    for symbol, mv in mv_row.items():
        if mv <= 0 or symbol not in prices.columns:
            continue
        price_used, _ = _price_on_or_before(prices[symbol].dropna(), start)
        if price_used and price_used > 0:
            shares[symbol] = mv / price_used
    return pd.Series(shares)


def build_positions_from_transactions(
    transactions: pd.DataFrame,
    prices: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    starting_positions: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if transactions.empty or prices.empty:
        return pd.DataFrame(), pd.DataFrame()

    tx = transactions.copy()
    tx["Date"] = pd.to_datetime(tx["Date"], errors="coerce")
    tx = tx.dropna(subset=["Date"])
    tx = tx[(tx["Date"] >= start) & (tx["Date"] <= end)]
    if tx.empty:
        return pd.DataFrame(), pd.DataFrame()

    tx["Quantity"] = tx.get("Quantity", pd.Series(dtype=float)).apply(_to_number)
    tx["Price"] = tx.get("Price", pd.Series(dtype=float)).apply(_to_number)
    tx["Amount"] = tx.get("Amount", pd.Series(dtype=float)).apply(_to_number)
    tx["Action"] = tx["Action"].astype(str)
    tx = tx[tx["Symbol"].notna()]

    tx = tx[tx["Action"].str.lower().isin(["buy", "sell"])]
    if tx.empty:
        return pd.DataFrame(), pd.DataFrame()

    debug_rows = []
    share_events = []
    for _, row in tx.iterrows():
        symbol = row["Symbol"]
        action = row["Action"].strip().lower()
        date = row["Date"]
        if symbol not in prices.columns:
            continue
        price_series = prices[symbol].dropna()
        price_used, source = _price_on_or_before(price_series, date)
        if price_used is None:
            warnings.warn(f"[WARN] Missing price for {symbol} around {date.date()}")
            continue

        qty = row["Quantity"]
        amount = row["Amount"]
        if pd.notna(qty) and qty != 0:
            shares = int(abs(qty))
            implied_fill = price_used
            confidence = True
        else:
            shares, implied_fill, confidence = infer_shares_from_amount(amount, price_used)

        sign = 1 if action == "buy" else -1
        share_events.append({
            "Date": date,
            "Symbol": symbol,
            "Shares": sign * shares,
        })
        debug_rows.append({
            "date": date.date().isoformat(),
            "symbol": symbol,
            "action": action.upper(),
            "amount": f"{amount:.2f}" if pd.notna(amount) else "n/a",
            "price_used": f"{price_used:.2f}",
            "implied_fill": f"{implied_fill:.2f}" if pd.notna(implied_fill) else "n/a",
            "inferred_shares": shares,
            "confidence": "high" if confidence else "low",
            "price_source": source,
        })

    if not share_events:
        return pd.DataFrame(), pd.DataFrame(debug_rows)

    events_df = pd.DataFrame(share_events)
    events_df["Date"] = pd.to_datetime(events_df["Date"])
    events_df = events_df.groupby(["Date", "Symbol"])["Shares"].sum().reset_index()

    idx = prices.loc[start:end].index
    positions = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    if starting_positions is not None and not starting_positions.empty:
        for symbol, shares in starting_positions.items():
            if symbol in positions.columns and shares > 0:
                positions.iloc[0, positions.columns.get_loc(symbol)] = shares
    for _, ev in events_df.iterrows():
        if ev["Date"] not in positions.index:
            # Align to prior trading day (never future).
            prior = positions.index[positions.index <= ev["Date"]]
            if prior.empty:
                continue
            date = prior[-1]
        else:
            date = ev["Date"]
        positions.loc[date, ev["Symbol"]] += ev["Shares"]

    positions = positions.cumsum()
    positions = positions.clip(lower=0.0)
    return positions, pd.DataFrame(debug_rows)


def compute_cycle_returns(
    transactions: pd.DataFrame,
    prices: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    if transactions.empty or prices.empty:
        return pd.Series(dtype=float)

    tx = transactions.copy()
    tx["Date"] = pd.to_datetime(tx["Date"], errors="coerce")
    tx = tx.dropna(subset=["Date"])
    tx = tx[(tx["Date"] >= start) & (tx["Date"] <= end)]
    tx["Quantity"] = tx.get("Quantity", pd.Series(dtype=float)).apply(_to_number)
    tx["Price"] = tx.get("Price", pd.Series(dtype=float)).apply(_to_number)
    tx["Amount"] = tx.get("Amount", pd.Series(dtype=float)).apply(_to_number)
    tx["Action"] = tx["Action"].astype(str)
    tx = tx[tx["Symbol"].notna()]
    tx = tx[tx["Action"].str.lower().isin(["buy", "sell"])]
    if tx.empty:
        return pd.Series(dtype=float)

    returns = {}
    for symbol in tx["Symbol"].unique():
        if symbol not in prices.columns:
            continue
        sym_tx = tx[tx["Symbol"] == symbol].sort_values("Date")
        lots = []
        total_cost = 0.0
        realized_pnl = 0.0
        for _, row in sym_tx.iterrows():
            action = row["Action"].strip().lower()
            date = row["Date"]
            qty = row["Quantity"]
            price_used, _ = _price_on_or_before(prices[symbol].dropna(), date)
            if price_used is None:
                continue
            if pd.notna(qty) and qty != 0:
                shares = int(abs(qty))
            else:
                shares, _, _ = infer_shares_from_amount(row["Amount"], price_used)
            if shares <= 0:
                continue
            if action == "buy":
                lots.append({"shares": shares, "price": price_used})
                total_cost += shares * price_used
            else:
                remaining = shares
                while remaining > 0 and lots:
                    lot = lots[0]
                    take = min(remaining, lot["shares"])
                    realized_pnl += take * (price_used - lot["price"])
                    lot["shares"] -= take
                    remaining -= take
                    if lot["shares"] == 0:
                        lots.pop(0)
        # Unrealized on remaining lots to end date.
        end_price, _ = _price_on_or_before(prices[symbol].dropna(), end)
        if end_price is not None:
            for lot in lots:
                realized_pnl += lot["shares"] * (end_price - lot["price"])
                total_cost += lot["shares"] * lot["price"]
        if total_cost > 0:
            returns[symbol] = realized_pnl / total_cost

    return pd.Series(returns)
