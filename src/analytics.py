import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# CONFIG
TRANSACTIONS_CSV = DATA_DIR / "schwab_transactions.csv"
PRICES_CSV = DATA_DIR / "historical_prices.csv"
OUTPUT_HOLDINGS_TS = DATA_DIR / "holdings_timeseries.csv"
OUTPUT_SUMMARY = DATA_DIR / "portfolio_summary.csv"

STARTING_CASH = 0.0
KNOWN_SPLITS = {
    ("FCEL", pd.Timestamp("2024-11-11")): 1/30  # 1-for-30 reverse split
}

# -----------------------
# Cleaning and Structure
# -----------------------

def safe_read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return pd.read_csv(path)

def classify_action(action: str, description: str) -> str:
    """Map Schwab Action + Description into canonical action types."""
    a = (str(action) + " " + str(description)).lower()
    raw_action = str(action)

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

    # Apply classification (CRITICAL STEP)
    df["ActionType"] = df.apply(lambda r: classify_action(r["Action"], r["Description"]), axis=1)

    return df

# -------------------
# Pipeline Functions
# -------------------

def build_positions_timeseries(transactions: pd.DataFrame, prices: pd.DataFrame, starting_cash: float = STARTING_CASH, known_splits: dict = KNOWN_SPLITS):
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
        for _, row in df_date.iterrows():
            typ = row.get("ActionType", "OTHER")
            sym = row.get("Symbol", np.nan)
            qty = float(row.get("Quantity", 0.0)) if pd.notna(row.get("Quantity")) else 0.0
            amt = float(row.get("Amount", 0.0)) if pd.notna(row.get("Amount")) else 0.0

            # Only actions affecting share count need an update here
            if typ in ["BUY", "DIVIDEND_REINVEST"]:
                if pd.notna(sym):
                    holdings[sym] = holdings.get(sym, 0.0) + qty
                cash_balance += amt # Negative amount for cash outflow/cost
            elif typ == "SELL":
                if pd.notna(sym):
                    holdings[sym] = holdings.get(sym, 0.0) - qty
                cash_balance += amt # Positive amount for cash inflow
            elif typ == "TRANSFER":
                if pd.isna(sym):
                    cash_balance += amt # Cash-only transfer
                else:
                    # Stock transfer
                    if amt < 0 or qty < 0:
                        holdings[sym] = holdings.get(sym, 0.0) - abs(qty)
                    else:
                        holdings[sym] = holdings.get(sym, 0.0) + abs(qty)
            elif typ == "CASH_DIVIDEND" or typ in ("FEE", "INTEREST"):
                cash_balance += amt
            elif typ in ["Stock Split"]:
                holdings[sym] = holdings.get(sym, 0.0) + qty
                cash_balance += amt
            elif typ in ["Reverse Split"]:
                if pd.notna(sym) and sym == "FCEL":
                    key = (sym, date)
                    if key in KNOWN_SPLITS:
                        split_ratio = KNOWN_SPLITS[key]
                        holdings[sym] = holdings.get(sym, 0.0) * float(split_ratio)
                        cash_balance += amt
                else:
                    pass
            else:
                cash_balance += amt # Default cash impact
        
        # Snapshot after processing this date
        snapshots.append((date, holdings.copy(), cash_balance))
    
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
            
    return holdings_df, cash_ts

def compute_portfolio_valuation(holdings_ts: pd.DataFrame, prices: pd.DataFrame, cash_ts: pd.Series):
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

def compute_performance_metrics(ts: pd.DataFrame, price_index_col="total_value"):
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
        "latest_value": latest,
        "first_value": first,
        "total_return": total_return,
        "cagr": cagr,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown
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
    summary["as_of"] = pd.Timestamp.today().strftime("%Y-%m-%d")
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
            summary[f"top_{i+1}"] = name
            summary[f"top_{i+1}_weight"] = float(w)

    print(f"Saving portfolio summary to {OUTPUT_SUMMARY}")
    pd.Series(summary).to_csv(OUTPUT_SUMMARY)

    print("Pipeline complete.")
    return valuation_df, summary

if __name__ == "__main__":
    run_pipeline()