import dash
from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from viz.plots import empty_figure
from analytics.portfolio import build_portfolio_timeseries, risk_free_warning

dash.register_page(__name__, path="/benchmarks", name="Benchmarks")

# -----------------------
# Load data
# -----------------------
HOLDINGS_TS = "data/holdings_timeseries.csv"
BENCHMARKS_CSV = "data/benchmark_prices.csv"

holdings_ts = pd.read_csv(HOLDINGS_TS, parse_dates=["Date"], index_col="Date").sort_index()
portfolio_ts = build_portfolio_timeseries()
bench = pd.read_csv(BENCHMARKS_CSV, parse_dates=["Date"], index_col="Date").sort_index()

# Optional risk-free (10y). If not present, use 0.
# If you later save treasury_10y as CSV, set this path to that file.
TREASURY_CSV = "data/treasury.csv"
rf_daily = None

try:
    rf = pd.read_csv(TREASURY_CSV)

    # 0) Normalize column names (strip whitespace)
    rf.columns = [c.strip() for c in rf.columns]

    # 1) Find date column robustly (Date, DATE, date, etc.)
    date_col = None
    for c in rf.columns:
        if c.strip().lower() == "date":
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"No Date/DATE column found in {TREASURY_CSV}. Columns: {rf.columns.tolist()}")

    # 2) Parse date + set index
    rf[date_col] = pd.to_datetime(rf[date_col], errors="coerce")
    rf = rf.dropna(subset=[date_col]).set_index(date_col).sort_index()

    # If duplicate dates exist, keep last (FRED sometimes has quirks)
    rf = rf[~rf.index.duplicated(keep="last")]

    # 3) Find the rate column robustly
    # Prefer known names; if not found, fallback to first numeric-ish column
    preferred = ["DGS10", "treasury_10y", "10Y", "10y", "value"]
    rf_col = next((c for c in preferred if c in rf.columns), None)

    if rf_col is None:
        # fallback: pick the first column that can be converted to numeric with at least some non-NaNs
        best = None
        best_non_na = -1
        for c in rf.columns:
            s = pd.to_numeric(rf[c], errors="coerce")
            non_na = s.notna().sum()
            if non_na > best_non_na:
                best_non_na = non_na
                best = c
        rf_col = best

    if rf_col is None:
        raise ValueError(f"No usable rate column found in {TREASURY_CSV}. Columns: {rf.columns.tolist()}")

    # 4) Clean rate series
    rf_series = pd.to_numeric(rf[rf_col], errors="coerce")

    # If your CSV has blanks (like 2010-01-01,), decide how to handle:
    # forward-fill is normal for yields; you could also dropna if you prefer.
    rf_series = rf_series.ffill()

    # 5) DGS10 is percent annual yield -> decimal annual yield
    rf_annual = rf_series / 100.0

    # 6) Convert annual yield to daily rate
    # Choose convention:
    DAYS_PER_YEAR = 252  # or 365 if you want calendar-day convention
    rf_daily = (1.0 + rf_annual).pow(1.0 / DAYS_PER_YEAR) - 1.0

except Exception as e:
    print(f"[WARN] Treasury rf_daily not loaded: {e}")
    rf_daily = None

# Available benchmarks
benchmark_options = [{"label": c, "value": c} for c in bench.columns]

# -----------------------
# Layout
# -----------------------
layout = html.Div([
    html.Br(),
    html.H2("Benchmarks & Performance"),
    html.Div(
        risk_free_warning(),
        style={"color": "#b45309", "marginBottom": "8px"},
    ) if risk_free_warning() else html.Div(),
    html.P(
        "Compare portfolio performance against benchmarks, track rolling excess return, "
        "and inspect beta/alpha relationships."
    ),

    html.Div([
        html.Label([
            "Select Benchmark(s)",
            html.Span(" (info)", id="benchmarks-info", style={"cursor": "pointer", "textDecoration": "underline"}),
        ]),
        dbc.Tooltip(
            "Benchmarks are used for cumulative return and rolling excess return calculations.",
            target="benchmarks-info",
            placement="right",
        ),
        dcc.Dropdown(
            id="bench-select",
            options=benchmark_options,
            value=[bench.columns[0]] if len(bench.columns) else [],
            multi=True
        ),
    ], style={"maxWidth": "650px"}),

    html.Br(),

    html.Div([
        html.Label("Select Date Range:"),
        dcc.DatePickerRange(
            id="bench-date-range",
            min_date_allowed=min(holdings_ts.index.min(), bench.index.min()),
            max_date_allowed=max(holdings_ts.index.max(), bench.index.max()),
            start_date=max(holdings_ts.index.min(), bench.index.min()),
            end_date=min(holdings_ts.index.max(), bench.index.max()),
        ),
    ]),

    html.Br(),

    html.Div([
        html.Label("Rolling Window (trading days):"),
        dcc.Slider(
            id="roll-window",
            min=20,
            max=252,
            step=5,
            value=63,
            marks={20: "20", 63: "63", 126: "126", 252: "252"},
            tooltip={"placement": "bottom", "always_visible": False}
        )
    ], style={"maxWidth": "650px"}),

    html.Br(), html.Br(),

    # 1) Cumulative Return
    dcc.Loading(dcc.Graph(id="cumret-graph")),

    html.Br(),

    # 2) Rolling Excess Return
    html.Div([
        html.Span("Rolling Excess Return", id="rolling-excess-info", style={"cursor": "pointer", "textDecoration": "underline"}),
        dbc.Tooltip(
            "Annualized average return difference between portfolio and benchmark over the rolling window.",
            target="rolling-excess-info",
            placement="right",
        ),
    ]),
    dcc.Loading(dcc.Graph(id="rolling-excess-graph")),

    html.Br(),

    # 2b) Tracking Error + Information Ratio
    html.Div([
        html.Span("Tracking Error & Information Ratio", id="tracking-info", style={"cursor": "pointer", "textDecoration": "underline"}),
        dbc.Tooltip(
            "Tracking error is annualized std dev of excess returns; info ratio is annualized excess return divided by tracking error.",
            target="tracking-info",
            placement="right",
        ),
    ]),
    dcc.Loading(dcc.Graph(id="tracking-error-graph")),

    html.Br(),

    # 3) Alpha & Beta (CAPM-style)
    html.Div([
        html.Span("Alpha & Beta (CAPM)", id="alpha-beta-info", style={"cursor": "pointer", "textDecoration": "underline"}),
        dbc.Tooltip(
            "CAPM regression of portfolio excess returns on benchmark excess returns (daily).",
            target="alpha-beta-info",
            placement="right",
        ),
    ]),
    dcc.Loading(dcc.Graph(id="alpha-beta-graph")),
])


# -----------------------
# Helpers
# -----------------------
def _align_series(start_date, end_date, selected_benchmarks):
    """
    Returns aligned price series for portfolio and benchmarks over the date range.
    Portfolio series uses total_value.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Portfolio
    if "total_value_clean_rf" in portfolio_ts.columns:
        pv = portfolio_ts["total_value_clean_rf"].copy()
    else:
        pv = holdings_ts["total_value_clean"].copy()
    pv = pv.loc[start:end]

    # Benchmarks (price series)
    if not selected_benchmarks:
        bm = pd.DataFrame(index=pv.index)
    else:
        bm = bench[selected_benchmarks].copy()
        bm = bm.loc[start:end]

    # Align on intersection of dates
    idx = pv.index
    if not bm.empty:
        idx = idx.intersection(bm.index)

    pv = pv.reindex(idx).dropna()
    bm = bm.reindex(idx).ffill().dropna()

    return pv, bm


def _daily_returns(price_series: pd.Series) -> pd.Series:
    return price_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def _daily_returns_df(price_df: pd.DataFrame) -> pd.DataFrame:
    return price_df.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


# -----------------------
# Callbacks
# -----------------------
@callback(
    Output("cumret-graph", "figure"),
    Output("rolling-excess-graph", "figure"),
    Output("tracking-error-graph", "figure"),
    Output("alpha-beta-graph", "figure"),
    Input("bench-select", "value"),
    Input("bench-date-range", "start_date"),
    Input("bench-date-range", "end_date"),
    Input("roll-window", "value"),
)
def update_benchmark_graphs(selected_benchmarks, start_date, end_date, window):
    pv, bm = _align_series(start_date, end_date, selected_benchmarks)

    if pv.empty:
        empty = empty_figure("No data available for selected range.")
        return empty, empty, empty, empty

    # --- 1) Cumulative Return Comparison ---
    cum_df = pd.DataFrame(index=pv.index)
    cum_df["Portfolio(clean)"] = (pv / pv.iloc[0]) - 1

    if not bm.empty:
        for c in bm.columns:
            cum_df[c] = (bm[c] / bm[c].iloc[0]) - 1

    fig_cum = px.line(
        cum_df,
        title="Cumulative Return Comparison",
        labels={"value": "Cumulative Return", "index": "Date", "variable": "Series"}
    )
    fig_cum.update_layout(height=450, legend_title_text="")
    fig_cum.update_yaxes(tickformat=".1%")

    # --- returns for later sections ---
    if bm.empty:
        # If no benchmarks selected, we can’t compute excess return or CAPM
        fig_roll = empty_figure("Select at least one benchmark to compute rolling excess returns.")
        fig_ab = empty_figure("Select at least one benchmark to compute alpha/beta.")
        fig_te = empty_figure("Select at least one benchmark to compute tracking error.")
        return fig_cum, fig_roll, fig_te, fig_ab

    pv_ret = _daily_returns(pv)
    bm_ret = _daily_returns_df(bm)

    # Choose a "primary" benchmark for excess return and CAPM regression:
    # If multiple selected, use the first one (simple, predictable behavior).
    primary_bench = selected_benchmarks[0]
    b = bm_ret[primary_bench].dropna()

    # Align returns
    idx_r = pv_ret.index.intersection(b.index)
    pv_ret = pv_ret.reindex(idx_r)
    b = b.reindex(idx_r)

    # Risk-free daily series (optional)
    if rf_daily is not None:
        rf_aligned = rf_daily.reindex(idx_r).ffill().fillna(0.0)
    else:
        rf_aligned = pd.Series(0.0, index=idx_r)

    # --- 2) Rolling Excess Return ---
    idx_r = pv_ret.index.intersection(bm_ret.index)
    pv_ret = pv_ret.reindex(idx_r)
    bm_ret = bm_ret.reindex(idx_r)

    # excess return series for each benchmark
    roll_df = pd.DataFrame(index=idx_r)

    for bench_name in selected_benchmarks:
        if bench_name not in bm_ret.columns:
            continue

        excess = (pv_ret - bm_ret[bench_name]).dropna()
        roll_excess = excess.rolling(window).mean() * 252  # annualized
        roll_df[f"Excess vs {bench_name}"] = roll_excess

    if roll_df.dropna(how="all").empty:
        fig_roll = empty_figure("Not enough data to compute rolling excess returns.")
        fig_te = empty_figure("Not enough data to compute tracking error.")
        fig_ab = empty_figure("Not enough data to compute CAPM regression.")
    else:
        fig_roll = px.line(
            roll_df,
            title=f"Rolling Excess Return (window={window} days, annualized)",
            labels={"value": "Annualized Excess Return", "index": "Date", "variable": ""}
        )
        fig_roll.update_layout(height=450)
        fig_roll.update_yaxes(tickformat=".1%")

        fig_te = make_subplots(specs=[[{"secondary_y": True}]])
        fig_te.update_layout(
            title=f"Tracking Error & Information Ratio (window={window} days)",
            height=450,
            legend_title_text="",
        )

        for bench_name in selected_benchmarks:
            if bench_name not in bm_ret.columns:
                continue
            excess = (pv_ret - bm_ret[bench_name]).dropna()
            if excess.empty:
                continue
            te = excess.rolling(window).std() * np.sqrt(252)
            ir = (excess.rolling(window).mean() * 252) / te
            fig_te.add_trace(
                go.Scatter(x=te.index, y=te.values, mode="lines", name=f"TE vs {bench_name}"),
                secondary_y=False,
            )
            fig_te.add_trace(
                go.Scatter(
                    x=ir.index,
                    y=ir.values,
                    mode="lines",
                    name=f"IR vs {bench_name}",
                    line={"dash": "dot"},
                ),
                secondary_y=True,
            )
        fig_te.update_yaxes(title_text="Tracking Error", tickformat=".1%", secondary_y=False)
        fig_te.update_yaxes(title_text="Information Ratio", secondary_y=True)

        # --- 3) Alpha & Beta (CAPM-style) ---
        # CAPM regression: (Rp - Rf) = alpha + beta * (Rb - Rf)
        idx_rf = pv_ret.index
        rf_aligned = rf_daily.reindex(idx_rf).ffill().fillna(0.0)

        y_all = (pv_ret - rf_aligned).dropna()

        if y_all.empty:
            fig_ab = empty_figure("No return data available for CAPM.")
            fig_ab.update_layout(height=450)
            return fig_cum, fig_roll, fig_te, fig_ab

        fig_ab = go.Figure()

        for bench_name in selected_benchmarks:
            if bench_name not in bm_ret.columns:
                continue

            x_raw = bm_ret[bench_name].dropna()
            x_all = (x_raw - rf_aligned.reindex(x_raw.index).ffill().fillna(0.0)).dropna()

            idx_ab = y_all.index.intersection(x_all.index)
            if len(idx_ab) < 30:
                continue

            x = x_all.reindex(idx_ab)
            y = y_all.reindex(idx_ab)

            x_vals = x.values
            y_vals = y.values
            x_mean = x_vals.mean()
            y_mean = y_vals.mean()
            denom = ((x_vals - x_mean) ** 2).sum()

            if denom == 0:
                continue

            beta = ((x_vals - x_mean) * (y_vals - y_mean)).sum() / denom
            alpha_daily = y_mean - beta * x_mean  # daily alpha (excess)

            # Scatter points
            fig_ab.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=f"{bench_name} points",
                opacity=0.45
            ))

            # Fitted line
            x_line = np.linspace(float(x.min()), float(x.max()), 100)
            y_line = alpha_daily + beta * x_line
            fig_ab.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"{bench_name} fit (β={beta:.2f}, α={alpha_daily*252:.2%})"
            ))

        fig_ab.update_layout(
            title="CAPM Scatter: Portfolio Excess vs Benchmark Excess (daily)",
            xaxis_title="Benchmark Excess Return (daily)",
            yaxis_title="Portfolio Excess Return (daily)",
            height=500,
            legend_title_text=""
        )
        fig_ab.update_xaxes(tickformat=".2%")
        fig_ab.update_yaxes(tickformat=".2%")

    return fig_cum, fig_roll, fig_te, fig_ab
