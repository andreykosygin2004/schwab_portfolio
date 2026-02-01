from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# CONFIG
TRANSACTIONS_CSV = DATA_DIR / "schwab_transactions.csv"
PRICES_CSV = DATA_DIR / "historical_prices.csv"
STOCK_SPLITS_CSV = DATA_DIR / "stock_splits.csv"
OUTPUT_HOLDINGS_TS = DATA_DIR / "holdings_timeseries.csv"
OUTPUT_SUMMARY = DATA_DIR / "portfolio_summary.csv"

STARTING_CASH = 0.0

SYMBOL_ALIAS = {
    "35952H601": "FCEL"
}

# -----------------------
# Loading
# -----------------------

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return pd.read_csv(path)

def load_splits(path: Path) -> dict[tuple[str, pd.Timestamp], float]:
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path, parse_dates=["date"])
    mapping = {}
    for _, r in df.iterrows():
        sym = str(r["Ticker"]).upper()
        dt = pd.to_datetime(r["date"]).normalize()
        mapping[(sym, dt)] = float(r["split_ratio"])
    return mapping

SPLIT_MAP = load_splits(STOCK_SPLITS_CSV)

# ---------------------------
# Cleaning and Classification
# ---------------------------

def classify_action(action: str, description: str) -> str:
    """Map Schwab Action + Description into canonical action types."""
    a = (str(action) + " " + str(description)).lower()
    raw_action = str(action).strip()

    if "reverse split" in a or raw_action.lower() == "reverse split":
        return "REVERSE_SPLIT"
    if "stock split" in a or ("split" in a and "reverse" not in a):
        return "SPLIT"
    
    if raw_action == "Buy":
        if "reinvest" in a or "drip" in a:
            return "DIVIDEND_REINVEST"
        return "BUY"

    if raw_action == "Sell":
        return "SELL"

    if raw_action in ("Cash Dividend", "Qualified Dividend", "Special Dividend"):
        if "reinvest" in a or "drip" in a:
            return "DIVIDEND_REINVEST"
        return "CASH_DIVIDEND"

    if raw_action == "MoneyLink Transfer":
        return "TRANSFER"

    if raw_action in ("Margin Interest", "Bank Interest"):
        return "INTEREST"
    
    if "fee" in a or raw_action in ("Fee", "Fees", "Fees & Comm"):
        return "FEE"

    return "OTHER"

def normalize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # Normalize column names
    rename_map = {}
    for c in df.columns:
        if c in ["Fees & Comm", "Fees", "Fee"] and "Fees" not in df.columns:
            rename_map[c] = "Fees"
        if c in ["Amount", "Net Amount", "Net Amount ($)"] and "Amount" not in df.columns:
            rename_map[c] = "Amount"
    df = df.rename(columns=rename_map)

    # String cleanup
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Date Parsing
    if "Date" in df.columns:
        extracted = df["Date"].astype(str).str.extract(r"(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{1,2}-\d{1,2})", expand=False)
        df["Date"] = pd.to_datetime(extracted, errors="coerce")
        df = df.dropna(subset=["Date"])

    # Numeric Cleanup
    def clean_money_series(s):
        s = s.astype(str).str.replace(r'\s+', '', regex=True)
        s = s.str.replace(r'^\((.*)\)$', r'-\1', regex=True)
        s = s.str.replace(r'[$,]', '', regex=True)
        s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
        return pd.to_numeric(s, errors="coerce")

    for col in ["Quantity", "Price", "Fees", "Amount"]:
        if col in df.columns:
            df[col] = clean_money_series(df[col])

    # Symbol Cleanup
    if "Symbol" in df.columns:
        df["Symbol"] = df["Symbol"].replace(["", "nan", "NaN", "N/A", None], np.nan)
        df["Symbol"] = df["Symbol"].astype(str).str.strip().replace({"nan": np.nan})
        df.loc[df["Symbol"].notna(), "Symbol"] = df.loc[df["Symbol"].notna(), "Symbol"].str.upper()

    # Apply classification
    if "Action" not in df.columns:
        df["Action"] = ""
    if "Description" not in df.columns:
        df["Description"] = ""
    df["ActionType"] = df.apply(lambda r: classify_action(r["Action"], r["Description"]), axis=1)

    return df

def compute_clean_cash_balance(transactions: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """
    Rebuild cash balance excluding negative MoneyLink Transfers (withdrawals),
    but keeping deposits (+ MoneyLink) and all other cash impacts (buys/sells/div/fees/interest).
    """
    tx = transactions.copy()

    # Ensure Date is datetime
    tx["Date"] = pd.to_datetime(tx["Date"], errors="coerce")
    tx = tx.dropna(subset=["Date"])

    # Ensure numeric Amount
    if "Amount" in tx.columns:
        tx["Amount"] = (
            tx["Amount"]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        )
        tx["Amount"] = pd.to_numeric(tx["Amount"], errors="coerce").fillna(0.0)
    else:
        tx["Amount"] = 0.0

    # Ensure ActionType exists
    if "ActionType" not in tx.columns:
        tx["ActionType"] = tx.apply(lambda r: classify_action(r.get("Action", ""), r.get("Description", "")), axis=1)

    # IMPORTANT: your classify_action returns "TRANSFER" for MoneyLink Transfer
    # so we must identify MoneyLink transfers via raw Action too (most robust)
    is_moneylink = tx["Action"].astype(str).str.strip().eq("MoneyLink Transfer")

    # Remove ONLY negative MoneyLink transfers (withdrawals)
    neg_withdraw_mask = is_moneylink & (tx["Amount"] < 0)
    tx.loc[neg_withdraw_mask, "Amount"] = 0.0

    # Recompute clean cash balance
    cash_clean = tx.groupby("Date")["Amount"].sum().cumsum()
    cash_clean = cash_clean.reindex(index).ffill().fillna(0.0)

    return cash_clean


def future_split_factor(sym: str, trade_date: pd.Timestamp, split_map: dict) -> float:
    """
    Convert raw Schwab quantities into *adjusted-share* quantities compatible with
    auto_adjust=True prices.

    If a stock has a forward split AFTER the trade_date, adjusted prices will be smaller
    by the split ratio, so we need MORE shares to keep MV consistent.
      - forward split ratio > 1  => factor multiplies up
      - reverse split ratio < 1  => factor multiplies down
    """
    sym = str(sym).upper()
    trade_date = pd.to_datetime(trade_date).normalize()

    factor = 1.0
    # Multiply all splits that occur AFTER the trade date
    for (ticker, split_date), ratio in split_map.items():
        if ticker == sym and split_date > trade_date:
            factor *= float(ratio)
    return factor

# -------------------
# Pipeline Functions
# -------------------

def build_positions_timeseries(
    transactions: pd.DataFrame,
    prices: pd.DataFrame,
    starting_cash: float = STARTING_CASH,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build daily holdings and cash balance using an explicit chronological simulation.
    Relies on ADJUSTED prices, therefore split transactions only affect cash flow, not holdings.
    """
    
    tr = transactions.sort_values("Date").reset_index(drop=True)
    
    # 1. Initialization
    all_tickers = sorted(set(prices.columns) | set(tr["Symbol"].dropna().unique()))
    holdings = {t: 0.0 for t in all_tickers}
    cash_balance = float(starting_cash)
    
    snapshots = []
    
    # 2. Simulation (Iterrows is retained for strict chronological cash flow and holdings state)
    for date, df_date in tr.groupby("Date"):
        date_norm = pd.to_datetime(date).normalize()
        for _, row in df_date.iterrows():
            typ = row.get("ActionType", "OTHER")
            raw_sym = row.get("Symbol", np.nan)

            # normalize symbol/alias
            sym = SYMBOL_ALIAS.get(raw_sym, raw_sym)
            sym = str(sym).upper() if pd.notna(sym) else np.nan

            qty = float(row.get("Quantity")) if pd.notna(row.get("Quantity")) else 0.0
            amt = float(row.get("Amount")) if pd.notna(row.get("Amount")) else 0.0

            # ---- Trades & reinvestments (adjust qty using future split factor) ----
            if typ in ("BUY", "DIVIDEND_REINVEST"):
                if pd.notna(sym) and sym != "NAN":
                    adj_factor = future_split_factor(sym, date_norm, SPLIT_MAP)
                    qty_adj = qty * adj_factor
                    holdings[sym] = holdings.get(sym, 0.0) + qty_adj
                # cash outflow usually negative already
                cash_balance += amt

            elif typ == "SELL":
                if pd.notna(sym) and sym != "NAN":
                    adj_factor = future_split_factor(sym, date_norm, SPLIT_MAP)
                    qty_adj = qty * adj_factor
                    holdings[sym] = holdings.get(sym, 0.0) - qty_adj
                # cash inflow usually positive already
                cash_balance += amt

            # ---- Transfers ----
            elif typ == "TRANSFER":
                # cash-only transfer
                if pd.isna(sym) or sym == "NAN":
                    cash_balance += amt
                else:
                    # stock transfer: if you ever have these, you likely want them to reflect share movement.
                    # Use qty sign if present; otherwise use amt sign.
                    if qty != 0:
                        # Assume qty is already share-count (Schwab transfer rows may be raw; apply split factor just in case)
                        adj_factor = future_split_factor(sym, date_norm, SPLIT_MAP)
                        qty_adj = qty * adj_factor
                        holdings[sym] = holdings.get(sym, 0.0) + qty_adj
                    else:
                        # no qty info, nothing to do for shares
                        pass
                    cash_balance += amt

            # ---- Cash flows ----
            elif typ == "CASH_DIVIDEND" or typ in ("FEE", "INTEREST"):
                cash_balance += amt

            # ---- Splits: IGNORE for holdings (handled via adjusted qty at trade time) ----
            elif typ in ("SPLIT", "REVERSE_SPLIT"):
                # Optional: cash-in-lieu sometimes appears here
                cash_balance += amt

            # ---- Default: apply any cash impact, do not change holdings ----
            else:
                cash_balance += amt
        
        # Snapshot after processing this date
        snapshots.append((date_norm, holdings.copy(), cash_balance))
    
    # 3. Vectorization and Reindexing
    if not snapshots:
        # Handles empty transaction history case
        empty_holdings = pd.DataFrame(0.0, index=prices.index, columns=all_tickers)
        empty_cash = pd.Series(starting_cash, index=prices.index)
        return empty_holdings, empty_cash

    snap_dates = [s[0] for s in snapshots]
    snap_holdings = [s[1] for s in snapshots]
    snap_cash = [s[2] for s in snapshots]

    holdings_df = pd.DataFrame(snap_holdings, index=pd.to_datetime(snap_dates))
    cash_series = pd.Series(snap_cash, index=pd.to_datetime(snap_dates))

    # Reindex to all price dates and forward-fill the daily state
    market_index = prices.index
    holdings_df = holdings_df.reindex(market_index).ffill().fillna(0.0)
    cash_ts = cash_series.reindex(market_index).ffill().fillna(starting_cash)

    # Ensure all price tickers exist in holdings_df
    for col in prices.columns:
        if col not in holdings_df.columns:
            holdings_df[col] = 0.0
    holdings_df = holdings_df[sorted(holdings_df.columns)]
            
    return holdings_df, cash_ts

def compute_portfolio_valuation(
    holdings_ts: pd.DataFrame,
    prices: pd.DataFrame,
    cash_ts: pd.Series,
) -> pd.DataFrame:
    """
    Multiply holdings by price (assumed Adjusted Close) to get market values, add cash balance.
    """
    # Align prices to holdings index (efficient pandas reindex)
    prices_aligned = prices.reindex(holdings_ts.index).ffill().fillna(0.0)
    
    # Ensure columns match for multiplication
    missing_price_cols = set(holdings_ts.columns) - set(prices_aligned.columns)
    if missing_price_cols:
        print(f"Warning: No price data for {missing_price_cols}. Assuming $0 value.")
        for col in missing_price_cols:
            prices_aligned[col] = 0.0
            
    prices_aligned = prices_aligned[holdings_ts.columns] # Match order

    # Vectorized calculation for market values
    market_vals = holdings_ts * prices_aligned
    market_vals.columns = [f"MV_{c}" for c in market_vals.columns]

    df = pd.concat([market_vals], axis=1)
    df["portfolio_value"] = df.sum(axis=1)
    
    # Ensure cash_ts is aligned and ready
    df["cash_balance"] = cash_ts.reindex(df.index).ffill().fillna(0.0)
    df["total_value"] = df["portfolio_value"] + df["cash_balance"]
    
    return df

def compute_performance_metrics(ts: pd.DataFrame, price_index_col: str = "total_value") -> dict[str, float]:
    """Compute return, risk, and drawdown metrics for a price/value series."""
    s = ts[price_index_col].dropna()
    if s.empty:
        return {}
    latest = float(s.iloc[-1])
    first = float(s.iloc[0]) if s.iloc[0] != 0 else np.nan
    total_return = (latest / first) - 1.0 if first and first != 0 else np.nan

    days = (s.index[-1] - s.index[0]).days
    years = days / 365.25 if days > 0 else np.nan
    cagr = (latest / first) ** (1 / years) - 1 if years and years > 0 and first and first > 0 else np.nan

    daily_rets = s.pct_change().dropna()
    annual_vol = float(daily_rets.std() * np.sqrt(252)) if not daily_rets.empty else np.nan
    sharpe = (daily_rets.mean() * 252) / annual_vol if (annual_vol is not None and not pd.isna(annual_vol)) else np.nan

    running_max = s.cummax()
    drawdown = (s - running_max) / running_max
    max_drawdown = float(drawdown.min())

    metrics = {
        "Latest Value": latest,
        "First Value": first,
        "Total Return": total_return,
        "CAGR": cagr,
        "Annual Vol": annual_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown
    }
    return metrics

# Main pipeline

def run_pipeline():
    print("Loading files...")
    transactions = safe_read_csv(TRANSACTIONS_CSV)
    
    # 1. Clean and Classify
    transactions = normalize_transactions(transactions)
    
    valid_actions = transactions[transactions["Action"].astype(str).str.strip() != ""]
    if valid_actions.empty:
        print("No valid transactions found. Exiting.")
        return
        
    first_txn_date = valid_actions["Date"].min()
    print(f"Start Date: {first_txn_date.date()}")
    
    # 2. Load Prices and Prepare Index
    prices = pd.read_csv(PRICES_CSV, parse_dates=["Date"]).set_index("Date").sort_index()
    
    # Prices must be cleaned for N/A values before simulation
    prices = prices.ffill().fillna(0.0)
    
    # Filter prices to start roughly when transactions start
    prices = prices[prices.index >= first_txn_date]

    print(f"Loaded {len(transactions)} transactions and price series from {prices.index.min().date()} to {prices.index.max().date()}.")

    print("Building holdings timeseries (Adjusted Price Logic)...")
    # This function now correctly ignores split transactions in holdings count.
    holdings_ts, cash_ts = build_positions_timeseries(transactions, prices, starting_cash=STARTING_CASH)

    print("Computing portfolio valuation...")
    valuation_df = compute_portfolio_valuation(holdings_ts, prices, cash_ts)
    cash_balance_clean = compute_clean_cash_balance(transactions, valuation_df.index)
    valuation_df["cash_balance_clean"] = cash_balance_clean
    valuation_df["total_value_clean"] = valuation_df["portfolio_value"] + valuation_df["cash_balance_clean"]
    
    # Trim valuation_df to start exactly at the first transaction date
    valuation_df = valuation_df.loc[valuation_df.index >= first_txn_date]

    print("Computing performance metrics and cumulative deposits...")
    metrics = compute_performance_metrics(valuation_df, price_index_col="total_value")

    # Recalculate cumulative deposits using the same classification logic
    capital_mask = (transactions["Amount"].notna()) & (~transactions["ActionType"].isin([
        "CASH_DIVIDEND", "DIVIDEND_REINVEST", "INTEREST", "FEE", "SPLIT", "REVERSE_SPLIT" # Exclude all forms of portfolio P&L
    ]))
    
    # Group by Date and apply cumulative sum
    # Grouped sum is done *before* reindexing for correctness
    cumulative_deposits = transactions.loc[capital_mask].groupby("Date")["Amount"].sum().cumsum()
    
    # Reindex to the final market date index
    cumulative_deposits = cumulative_deposits.reindex(valuation_df.index).ffill().fillna(0.0)
    
    valuation_df["cumulative_deposits"] = cumulative_deposits
    
    # --- Output and Summary (Unchanged) ---
    print(f"Saving holdings timeseries to {OUTPUT_HOLDINGS_TS}")
    valuation_df.to_csv(OUTPUT_HOLDINGS_TS)

    # Prepare summary
    summary = metrics.copy()
    summary["As of"] = pd.Timestamp.today().strftime("%Y-%m-%d")
    # top holdings
    mv_cols = [c for c in valuation_df.columns if c.startswith("MV_")]
    if mv_cols:
        latest_market_vals = valuation_df[mv_cols].iloc[-1]
        total_mv = latest_market_vals.sum()
        if total_mv > 0:
            weights = latest_market_vals / total_mv
        else:
            weights = latest_market_vals * 0.0
        sorted_weights = weights.sort_values(ascending=False)
        for i, (col, w) in enumerate(sorted_weights.items()):
            if i >= 5:
                break
            name = col.replace("MV_", "")
            summary[f"Top {i+1}"] = name
            summary[f"Top {i+1} Weight"] = float(w)

    print(f"Saving portfolio summary to {OUTPUT_SUMMARY}")
    pd.Series(summary).to_csv(OUTPUT_SUMMARY)

    print("Pipeline complete.")
    return valuation_df, summary

if __name__ == "__main__":
    run_pipeline()
