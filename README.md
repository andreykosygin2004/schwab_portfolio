# Schwab Portfolio Dashboard

Personal investing dashboard built with Dash (Plotly). The app combines portfolio performance, benchmark comparisons, macro drivers, regimes, and attribution into a single, interview-ready interface.

## Screenshots

<img width="1707" height="815" alt="Screenshot 2026-01-29 at 7 50 57 PM" src="https://github.com/user-attachments/assets/60f6130f-5687-4d4a-b4e2-5f5c3ef16310" />

<img width="1710" height="894" alt="Screenshot 2026-01-29 at 7 53 02 PM" src="https://github.com/user-attachments/assets/ec3f6ded-8562-414a-a1a7-949da1cc3f5d" />

<img width="1710" height="944" alt="Screenshot 2026-01-29 at 7 53 50 PM" src="https://github.com/user-attachments/assets/20a69371-e2f9-4005-a805-5677efb11de5" />

<img width="1709" height="916" alt="Screenshot 2026-01-29 at 8 00 41 PM" src="https://github.com/user-attachments/assets/dddcf7bd-c82f-4502-925a-6cb65f74fb6f" />

<img width="1710" height="749" alt="Screenshot 2026-02-05 at 10 01 43 AM" src="https://github.com/user-attachments/assets/013d22ed-e9cd-46cd-97ed-ada837708885" />

## Highlights
- Portfolio overview with clean cash-adjusted totals and holdings time series.
- Benchmark comparisons (cumulative returns, excess return, CAPM diagnostics).
- Macro dashboard with regime metrics, rates/credit/commodities proxies.
- Risk & drawdowns, VaR/CVaR, rolling risk metrics.
- Factor attribution and regime analysis.
- Factor rotation sleeve (monthly, blended weights) and trade blotter.

Note: The Alpha Holdings Tracker page remains a work-in-progress; logic and labels are being refined. To provide more representative analytics, select most pages to begin on 08/01/2024, when portfolio activity and diversification became more meaningful.

## Setup
Clone repo: https://github.com/andreykosygin2004/schwab_portfolio.git
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
