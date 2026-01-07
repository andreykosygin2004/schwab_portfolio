import dash
from dash import html, dcc, Input, Output, callback, dash_table, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
from analytics import compute_performance_metrics
from analytics.portfolio import build_portfolio_timeseries, risk_free_warning
from viz.plots import empty_figure

dash.register_page(__name__, path="/", name="Home")

prices = pd.read_csv("data/historical_prices.csv", index_col=0, parse_dates=True)
tickers = prices.columns.tolist()

holdings_ts = pd.read_csv("data/holdings_timeseries.csv", parse_dates=["Date"], index_col="Date").sort_index()
portfolio_ts = build_portfolio_timeseries()
mv_options = [{"label": "Portfolio Total", "value": "Portfolio Total"}] + [
    {"label": t, "value": t} for t in tickers]

INFO_STYLE = {"cursor": "pointer", "textDecoration": "underline"}
GRAPH_HEIGHT = 450
ROLL_VOL_WINDOW = 63

min_date = holdings_ts.index.min().date().isoformat()
max_date = holdings_ts.index.max().date().isoformat()

layout = html.Div([
    html.Br(),
    html.H2("Portfolio Overview"),
    html.Div(
        risk_free_warning(),
        style={"color": "#b45309", "marginBottom": "8px"},
    ) if risk_free_warning() else html.Div(),
    html.P(
        "Track total portfolio value, clean cash-adjusted value, and individual holdings "
        "over the selected window."
    ),

    html.Br(),
    html.H3("Holdings Market Values & Portfolio Total"),
    html.Label("Select Holdings:"),
    dcc.Dropdown(
        id="mv-ticker-select",
        options=mv_options,
        value=["Portfolio Total"],
        multi=True
    ),
    html.Label("Select Date Range:"),
    dcc.DatePickerRange(
        id="mv-date-range",
        min_date_allowed=min_date,
        max_date_allowed=max_date,
        start_date=min_date,
        end_date=max_date
    ),
    dcc.Loading(dcc.Graph(id="mv-graph")),

    html.Br(),
    html.H3("Portfolio Summary", style={"textAlign": "center"}),
    html.Br(),
    dash_table.DataTable(
        id="mv-summary-table",
        columns=[
            {"name": "Metric", "id": "Metric"},
            {"name": "Value", "id": "Value"}
        ],
        data=[],  # filled by callback
        style_cell={"textAlign": "left", "padding": "8px"},
        style_header={"fontWeight": "bold"},
        style_table={"width": "60%", "margin": "auto"}
    ),

    html.Hr(),
    html.H3("Portfolio Total (Excluding Negative Cash Transfers - Cash Earns Risk-Free)"),
    html.Label("Select Date Range:"),
    dcc.DatePickerRange(
        id="pv-clean-date-range",
        min_date_allowed=holdings_ts.index.min(),
        max_date_allowed=holdings_ts.index.max(),
        start_date=holdings_ts.index.min(),
        end_date=holdings_ts.index.max()
    ),
    dcc.Loading(dcc.Graph(id="pv-clean-graph")),

    html.Br(),
    html.H3("Portfolio Summary", style={"textAlign": "center"}),
    html.Br(),
    dash_table.DataTable(
    id="pv-clean-summary-table",
    columns=[
        {"name": "Metric", "id": "Metric"},
        {"name": "Value", "id": "Value"},
    ],
    data=[],
    style_cell={"textAlign": "left", "padding": "8px"},
    style_header={"fontWeight": "bold"},
    style_table={"width": "60%", "margin": "auto"}
    ),
    
    html.Hr(),
    html.Br(),
    html.H3([
        f"Rolling Portfolio Volatility (Baseline, {ROLL_VOL_WINDOW}D)",
        html.Span(" (info)", id="rolling-vol-info", style=INFO_STYLE),
    ]),
    dbc.Tooltip(
        "Annualized rolling volatility based on daily portfolio returns.",
        target="rolling-vol-info",
        placement="right",
    ),
    dcc.Loading(dcc.Graph(id="pv-rolling-vol-graph")
                ),

    html.Hr(),
    html.Br(),
    html.H3("General Overview (Price History)"),
    html.Label("Select Tickers:"),
    dcc.Dropdown(
        id="ticker-select",
        options=[{"label": t, "value": t} for t in tickers],
        value=[],
        multi=True
    ),
    html.Br(),
    html.Label("Select Date Range:"),
    dcc.DatePickerRange(id="date-range",
                        min_date_allowed=prices.index.min(),
                        max_date_allowed=prices.index.max(),
                        start_date=prices.index.min(),
                        end_date=prices.index.max()
    ),
    html.Br(), html.Br(),
    dcc.Loading(dcc.Graph(id="ticker-graph"))
])

def fmt_metric_value(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, (int, float)):
        return f"{v:,.6f}"
    return str(v)

def metrics_dict_to_rows(metrics: dict):
    # Optional: friendlier labels + ordering
    order = [
        "Latest Value",
        "First Value",
        "Total Return",
        "CAGR",
        "Annual Vol",
        "Sharpe Ratio",
        "Max Drawdown",
    ]
    labels = {k: k for k in order}

    rows = []
    for k in order:
        if k in metrics:
            rows.append({"Metric": labels.get(k, k), "Value": fmt_metric_value(metrics[k])})

    # Include any extra keys not in `order`
    for k, v in metrics.items():
        if k not in order:
            rows.append({"Metric": labels.get(k, k), "Value": fmt_metric_value(v)})

    return rows


@callback(
    Output("mv-graph", "figure"),
    Input("mv-ticker-select", "value"),
    Input("mv-date-range", "start_date"),
    Input("mv-date-range", "end_date")
)
def update_mv_graph(selected_holdings, start_date, end_date):
    if not selected_holdings:
        return empty_figure("No holdings selected", height=GRAPH_HEIGHT)

    df_slice = holdings_ts.loc[start_date:end_date]
    portfolio_slice = portfolio_ts.loc[start_date:end_date]
    if df_slice.empty:
        return empty_figure("No data in selected range", height=GRAPH_HEIGHT)

    show_portfolio_total = "Portfolio Total" in selected_holdings

    # Build MV columns for selected tickers only
    selected_tickers = [t for t in selected_holdings if t != "Portfolio Total"]
    mv_cols = [f"MV_{t}" for t in selected_tickers if f"MV_{t}" in holdings_ts.columns]

    if show_portfolio_total:
        # IMPORTANT: portfolio total should include cash, so use total_value
        out = pd.DataFrame(index=df_slice.index)
        if "total_value_rf" in portfolio_slice.columns:
            out["Portfolio Total"] = portfolio_slice["total_value_rf"]
        else:
            out["Portfolio Total"] = df_slice["total_value"]

        # Optional: if you also want the selected tickers plotted alongside the total
        if mv_cols:
            out = out.join(df_slice[mv_cols])
        df = out
    else:
        if not mv_cols:
            return empty_figure("No valid tickers selected", height=GRAPH_HEIGHT)
        df = df_slice[mv_cols].copy()

    fig = px.line(df, title="Market Values / Portfolio Total")
    fig.update_layout(legend_title_text="Series", height=GRAPH_HEIGHT)
    fig.update_yaxes(title_text="Market Value ($)")
    fig.update_xaxes(title_text="Date")

    return fig


@callback(
    Output("mv-summary-table", "data"),
    Input("mv-ticker-select", "value"),
    Input("mv-date-range", "start_date"),
    Input("mv-date-range", "end_date"),
)
def update_portfolio_summary_table(selected_holdings, start_date, end_date):
    # Only refresh table when Portfolio Total is selected
    if not selected_holdings or "Portfolio Total" not in selected_holdings:
        return no_update

    df_slice = holdings_ts.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)].copy()
    if df_slice.empty:
        return [{"Metric": "error", "Value": "No data in selected range"}]

    # 1) Base performance metrics from total_value
    if "total_value_rf" in portfolio_ts.columns:
        total_series = portfolio_ts.loc[start_date:end_date, "total_value_rf"].dropna()
        metrics = compute_performance_metrics(total_series.to_frame("total_value_rf"), price_index_col="total_value_rf")
    else:
        metrics = compute_performance_metrics(df_slice, price_index_col="total_value")
    rows = metrics_dict_to_rows(metrics)

    # 2) Add Top holdings + weights (based on latest MV_ row in the selected range)
    mv_cols = [c for c in df_slice.columns if c.startswith("MV_")]
    if mv_cols:
        latest_mvs = df_slice[mv_cols].iloc[-1].fillna(0.0)
        total_mv = float(latest_mvs.sum())

        rows.append({"Metric": "", "Value": ""})  # spacer row
        rows.append({"Metric": "Top Holdings (end of range)", "Value": ""})

        if total_mv > 0:
            weights = (latest_mvs / total_mv).sort_values(ascending=False)
            for i, (mv_col, w) in enumerate(weights.items(), start=1):
                if i > 5:
                    break
                ticker = mv_col.replace("MV_", "")
                rows.append({"Metric": f"Top {i}", "Value": ticker})
                rows.append({"Metric": f"Top {i} Weight", "Value": f"{w:.4%}"})
        else:
            rows.append({"Metric": "Top holdings", "Value": "No market value in range"})

    return rows

@callback(
    Output("pv-clean-graph", "figure"),
    Output("pv-clean-summary-table", "data"),
    Input("pv-clean-date-range", "start_date"),
    Input("pv-clean-date-range", "end_date"),
)
def update_pv_clean_graph_and_metrics(start_date, end_date):
    # Slice by selected range (DatePickerRange gives strings)
    df = holdings_ts.loc[start_date:end_date].copy()
    portfolio_slice = portfolio_ts.loc[start_date:end_date]

    # -------- Graph --------
    if df.empty:
        fig = empty_figure("No data in selected range", height=GRAPH_HEIGHT)
        return fig, []

    series_name = "total_value_clean_rf" if "total_value_clean_rf" in portfolio_slice.columns else "total_value_clean"
    fig = px.line(
        portfolio_slice,
        y=series_name,
        title="Portfolio Value (Clean)",
    )
    fig.update_layout(
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Date",
        height=GRAPH_HEIGHT
    )

    # -------- Metrics --------
    if "total_value_clean_rf" in portfolio_slice.columns:
        metrics_clean = compute_performance_metrics(
            portfolio_slice[[series_name]],
            price_index_col=series_name,
        )
    else:
        metrics_clean = compute_performance_metrics(df, price_index_col="total_value_clean")
    rows = metrics_dict_to_rows(metrics_clean)

    return fig, rows


@callback(
    Output("ticker-graph", "figure"),
    Input("ticker-select", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date")
)
def update_ticker_graph(selected_tickers, start_date, end_date):
    if not selected_tickers:
        return empty_figure("No tickers selected", height=GRAPH_HEIGHT)
    
    df = prices.loc[start_date:end_date, selected_tickers]
    if df.dropna(how="all").empty:
        return empty_figure("No data in selected range", height=GRAPH_HEIGHT)

    fig = px.line(df, title="Selected Tickers")
    fig.update_layout(legend_title_text="Tickers", height=GRAPH_HEIGHT)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price")

    return fig


@callback(
    Output("pv-rolling-vol-graph", "figure"),
    Input("pv-clean-date-range", "start_date"),
    Input("pv-clean-date-range", "end_date"),
)
def update_rolling_vol_graph(start_date, end_date):
    df = holdings_ts.loc[start_date:end_date].copy()
    if df.empty:
        return empty_figure("No data in selected range", height=GRAPH_HEIGHT)
    series_name = "total_value_clean_rf" if "total_value_clean_rf" in portfolio_ts.columns else "total_value_clean"
    s = portfolio_ts.loc[start_date:end_date, series_name].dropna()
    if s.empty or len(s) < ROLL_VOL_WINDOW + 1:
        return empty_figure("Not enough data for rolling volatility", height=GRAPH_HEIGHT)
    ret = s.pct_change().dropna()
    vol = ret.rolling(ROLL_VOL_WINDOW).std() * np.sqrt(252)
    fig = px.line(vol, title=f"Rolling Volatility (Baseline, {ROLL_VOL_WINDOW}D)")
    fig.update_layout(height=GRAPH_HEIGHT, legend_title_text="")
    fig.update_yaxes(title_text="Annualized Volatility", tickformat=".1%")
    fig.update_xaxes(title_text="Date")
    return fig