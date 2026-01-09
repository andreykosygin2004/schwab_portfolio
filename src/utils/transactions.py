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


def build_positions_from_transactions(
    transactions: pd.DataFrame,
    prices: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
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
