# Schwab Portfolio Dashboard

Personal investing dashboard built with Dash (Plotly). The app combines portfolio performance, benchmark comparisons, macro drivers, regimes, and attribution into a single, interview-ready interface.

## Highlights
- Portfolio overview with clean cash-adjusted totals and holdings time series.
- Benchmark comparisons (cumulative returns, excess return, CAPM diagnostics).
- Macro dashboard with regime metrics, rates/credit/commodities proxies.
- Risk & drawdowns, VaR/CVaR, rolling risk metrics.
- Factor attribution and regime analysis.
- Factor rotation sleeve (monthly, blended weights) and trade blotter.

Note: The Alpha Holdings Tracker page remains a work-in-progress; logic and labels are being refined. To provide more representative analytics, select most pages to begin on 08/01/2024, when portfolio activity and diversification became more meaningful.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python src/dashboard.py
```
Then open http://127.0.0.1:8050 in your browser.

## Data notes
This project uses public market data via yfinance and FRED.

Portfolio holdings are loaded from a local CSV file.  
No personal brokerage data is included in this repository.

By default, the app runs using a hypothetical sample portfolio
(`data/hypothetical_portfolio.csv`).

Users may replace this file with their own portfolio data
using the same column format.

Other notes:
- Local data files live in `data/`.
- Price history is cached in `data/macro_cache/`.
- The app uses clean cash logic (negative MoneyLink transfers excluded).

## Pages
- Portfolio Overview
- Benchmark Comparison
- Portfolio Risk
- Holdings Risk
- Return Breakdown
- Factor Attribution
- Macro Analysis
- Regime Analysis
- Regime Forecast
- Factor Rotation
