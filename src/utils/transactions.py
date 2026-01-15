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


def _split_map(splits: pd.DataFrame | None) -> dict[str, list[tuple[pd.Timestamp, float]]]:
    if splits is None or splits.empty:
        return {}
    df = splits.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["Ticker"] = df["Ticker"].astype(str)
    out: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    for _, row in df.iterrows():
        out.setdefault(row["Ticker"], []).append((row["date"], float(row["split_ratio"])))
    for key in out:
        out[key] = sorted(out[key], key=lambda x: x[0])
    return out


def _cumulative_split_ratio(
    split_events: list[tuple[pd.Timestamp, float]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    ratio = 1.0
    for dt, split_ratio in split_events:
        if start < dt <= end:
            ratio *= split_ratio
    return ratio


def build_positions_from_transactions(
    transactions: pd.DataFrame,
    prices: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    starting_positions: pd.Series | None = None,
    splits: pd.DataFrame | None = None,
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
    split_events = _split_map(splits)
    for _, row in tx.iterrows():
        symbol = row["Symbol"]
        action = row["Action"].strip().lower()
        date = row["Date"]
        if symbol not in prices.columns:
            continue
        price_series = prices[symbol].dropna()
        factor_end = _cumulative_split_ratio(split_events.get(symbol, []), date, end)
        price_used = row["Price"]
        source = "transaction"
        if pd.isna(price_used):
            adj_price, source = _price_on_or_before(price_series, date)
            if adj_price is None:
                warnings.warn(f"[WARN] Missing price for {symbol} around {date.date()}")
                continue
            price_used = float(adj_price) * factor_end

        qty = row["Quantity"]
        amount = row["Amount"]
        use_qty = False
        if pd.notna(qty) and qty != 0:
            est_amount = abs(qty) * price_used
            if pd.notna(amount) and est_amount > 0:
                if abs(est_amount - abs(amount)) / est_amount <= 0.05:
                    use_qty = True
            else:
                use_qty = True
        if use_qty:
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
    split_events = _split_map(splits)
    if split_events:
        for symbol, events in split_events.items():
            if symbol not in positions.columns:
                continue
            for split_date, ratio in events:
                if ratio == 1 or pd.isna(ratio):
                    continue
                idx_dates = positions.index[positions.index <= split_date]
                if idx_dates.empty:
                    continue
                adj_date = idx_dates[-1]
                positions.loc[adj_date:, symbol] = positions.loc[adj_date:, symbol] * ratio
    positions = positions.clip(lower=0.0)
    return positions, pd.DataFrame(debug_rows)


def compute_cycle_returns(
    transactions: pd.DataFrame,
    prices: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    splits: pd.DataFrame | None = None,
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
    split_events = _split_map(splits)
    for symbol in tx["Symbol"].unique():
        if symbol not in prices.columns:
            continue
        sym_tx = tx[tx["Symbol"] == symbol].sort_values("Date")
        lots = []
        lot_multiples = []
        last_date: pd.Timestamp | None = None
        events = split_events.get(symbol, [])

        def apply_splits(to_date: pd.Timestamp) -> None:
            nonlocal last_date, lots
            if not events:
                last_date = to_date
                return
            if last_date is None:
                last_date = to_date
                return
            for dt, ratio in events:
                if last_date < dt <= to_date:
                    for lot in lots:
                        lot["shares"] *= ratio
                        lot["cost_per_share"] /= ratio
            last_date = to_date

        for _, row in sym_tx.iterrows():
            action = row["Action"].strip().lower()
            date = row["Date"]
            qty = row["Quantity"]
            if last_date is None:
                last_date = date
            apply_splits(date)
            price_used = row["Price"]
            if pd.isna(price_used):
                price_used, _ = _price_on_or_before(prices[symbol].dropna(), date)
                if price_used is not None:
                    factor_end = _cumulative_split_ratio(events, date, end)
                    price_used = float(price_used) * factor_end
            if price_used is None or pd.isna(price_used):
                continue
            use_qty = False
            if pd.notna(qty) and qty != 0:
                est_amount = abs(qty) * price_used
                if pd.notna(row["Amount"]) and est_amount > 0:
                    if abs(est_amount - abs(row["Amount"])) / est_amount <= 0.05:
                        use_qty = True
                else:
                    use_qty = True
            if use_qty:
                shares = int(abs(qty))
            else:
                shares, _, _ = infer_shares_from_amount(row["Amount"], price_used)
            if shares <= 0:
                continue
            if action == "buy":
                lots.append({
                    "shares": shares,
                    "price": price_used,
                    "date": date,
                    "cost": shares * price_used,
                    "cost_per_share": price_used,
                    "realized_pnl": 0.0,
                })
            else:
                remaining = shares
                proceeds_per_share = price_used
                if pd.notna(row["Amount"]) and shares > 0:
                    proceeds_per_share = abs(row["Amount"]) / shares
                while remaining > 0 and lots:
                    lot = lots[0]
                    take = min(remaining, lot["shares"])
                    lot["realized_pnl"] += take * (proceeds_per_share - lot["cost_per_share"])
                    lot["shares"] -= take
                    remaining -= take
                    if lot["shares"] == 0:
                        lot_cost = lot["cost"]
                        if lot_cost > 0:
                            # Per-lot multiple so sequential cycles compound.
                            lot_multiples.append(1.0 + (lot["realized_pnl"] / lot_cost))
                        lots.pop(0)
        # Unrealized on remaining lots to end date.
        if last_date is not None:
            apply_splits(end)
        end_price, _ = _price_on_or_before(prices[symbol].dropna(), end)
        if end_price is not None:
            for lot in lots:
                lot["realized_pnl"] += lot["shares"] * (end_price - lot["cost_per_share"])
                lot_cost = lot["cost"]
                if lot_cost > 0:
                    lot_multiples.append(1.0 + (lot["realized_pnl"] / lot_cost))
        if lot_multiples:
            returns[symbol] = float(np.prod(lot_multiples) - 1.0)

    return pd.Series(returns)


def compute_trade_pnl(
    transactions: pd.DataFrame,
    prices: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    splits: pd.DataFrame | None = None,
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

    pnl = {}
    split_events = _split_map(splits)
    for symbol in tx["Symbol"].unique():
        if symbol not in prices.columns:
            continue
        sym_tx = tx[tx["Symbol"] == symbol].sort_values("Date")
        lots = []
        last_date: pd.Timestamp | None = None
        events = split_events.get(symbol, [])

        def apply_splits(to_date: pd.Timestamp) -> None:
            nonlocal last_date, lots
            if not events:
                last_date = to_date
                return
            if last_date is None:
                last_date = to_date
                return
            for dt, ratio in events:
                if last_date < dt <= to_date:
                    for lot in lots:
                        lot["shares"] *= ratio
                        lot["cost_per_share"] /= ratio
            last_date = to_date

        for _, row in sym_tx.iterrows():
            action = row["Action"].strip().lower()
            date = row["Date"]
            qty = row["Quantity"]
            if last_date is None:
                last_date = date
            apply_splits(date)
            price_used = row["Price"]
            if pd.isna(price_used):
                price_used, _ = _price_on_or_before(prices[symbol].dropna(), date)
                if price_used is not None:
                    factor_end = _cumulative_split_ratio(events, date, end)
                    price_used = float(price_used) * factor_end
            if price_used is None or pd.isna(price_used):
                continue
            use_qty = False
            if pd.notna(qty) and qty != 0:
                est_amount = abs(qty) * price_used
                if pd.notna(row["Amount"]) and est_amount > 0:
                    if abs(est_amount - abs(row["Amount"])) / est_amount <= 0.05:
                        use_qty = True
                else:
                    use_qty = True
            if use_qty:
                shares = int(abs(qty))
            else:
                shares, _, _ = infer_shares_from_amount(row["Amount"], price_used)
            if shares <= 0:
                continue
            if action == "buy":
                lots.append({
                    "shares": shares,
                    "cost_per_share": price_used,
                })
            else:
                remaining = shares
                proceeds_per_share = price_used
                if pd.notna(row["Amount"]) and shares > 0:
                    proceeds_per_share = abs(row["Amount"]) / shares
                while remaining > 0 and lots:
                    lot = lots[0]
                    take = min(remaining, lot["shares"])
                    pnl.setdefault(symbol, 0.0)
                    pnl[symbol] += take * (proceeds_per_share - lot["cost_per_share"])
                    lot["shares"] -= take
                    remaining -= take
                    if lot["shares"] == 0:
                        lots.pop(0)

        if last_date is not None:
            apply_splits(end)
        end_price, _ = _price_on_or_before(prices[symbol].dropna(), end)
        if end_price is not None:
            for lot in lots:
                pnl.setdefault(symbol, 0.0)
                pnl[symbol] += lot["shares"] * (end_price - lot["cost_per_share"])

    return pd.Series(pnl)


def compute_trade_summary(
    transactions: pd.DataFrame,
    prices: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    splits: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if transactions.empty or prices.empty:
        return pd.DataFrame()

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
        return pd.DataFrame()

    split_events = _split_map(splits)
    records = []
    for symbol in tx["Symbol"].unique():
        if symbol not in prices.columns:
            continue
        sym_tx = tx[tx["Symbol"] == symbol].sort_values("Date")
        lots = []
        invested = 0.0
        realized = 0.0
        last_date: pd.Timestamp | None = None
        events = split_events.get(symbol, [])

        def apply_splits(to_date: pd.Timestamp) -> None:
            nonlocal last_date, lots
            if not events:
                last_date = to_date
                return
            if last_date is None:
                last_date = to_date
                return
            for dt, ratio in events:
                if last_date < dt <= to_date:
                    for lot in lots:
                        lot["shares"] *= ratio
                        lot["cost_per_share"] /= ratio
            last_date = to_date

        for _, row in sym_tx.iterrows():
            action = row["Action"].strip().lower()
            date = row["Date"]
            qty = row["Quantity"]
            if last_date is None:
                last_date = date
            apply_splits(date)
            price_used = row["Price"]
            if pd.isna(price_used):
                price_used, _ = _price_on_or_before(prices[symbol].dropna(), date)
                if price_used is not None:
                    factor_end = _cumulative_split_ratio(events, date, end)
                    price_used = float(price_used) * factor_end
            if price_used is None or pd.isna(price_used):
                continue
            use_qty = False
            if pd.notna(qty) and qty != 0:
                est_amount = abs(qty) * price_used
                if pd.notna(row["Amount"]) and est_amount > 0:
                    if abs(est_amount - abs(row["Amount"])) / est_amount <= 0.05:
                        use_qty = True
                else:
                    use_qty = True
            if use_qty:
                shares = int(abs(qty))
            else:
                shares, _, _ = infer_shares_from_amount(row["Amount"], price_used)
            if shares <= 0:
                continue

            if action == "buy":
                cash_out = abs(row["Amount"]) if pd.notna(row["Amount"]) else shares * price_used
                invested += cash_out
                lots.append({"shares": shares, "cost_per_share": cash_out / shares})
            else:
                cash_in = abs(row["Amount"]) if pd.notna(row["Amount"]) else shares * price_used
                realized += cash_in
                remaining = shares
                while remaining > 0 and lots:
                    lot = lots[0]
                    take = min(remaining, lot["shares"])
                    lot["shares"] -= take
                    remaining -= take
                    if lot["shares"] == 0:
                        lots.pop(0)

        if last_date is not None:
            apply_splits(end)
        end_price, _ = _price_on_or_before(prices[symbol].dropna(), end)
        unrealized = 0.0
        if end_price is not None:
            for lot in lots:
                unrealized += lot["shares"] * end_price

        if invested > 0:
            pnl = (realized + unrealized) - invested
            total_return = pnl / invested
            records.append({
                "symbol": symbol,
                "invested": invested,
                "realized": realized,
                "unrealized": unrealized,
                "pnl": pnl,
                "total_return": total_return,
            })

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).set_index("symbol")
