import dash
from dash import html, dcc, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analytics.risk import (
    annualized_vol,
    compute_returns,
    drawdown_series,
    drawdown_episodes,
    max_drawdown,
    rolling_var,
    var_cvar,
)
from analytics_macro import load_portfolio_series, load_ticker_prices
from analytics.portfolio import risk_free_warning
from viz.plots import empty_figure

dash.register_page(__name__, path="/risk", name="Risk & Drawdowns")

PORTFOLIO_SERIES = load_portfolio_series()
MIN_DATE = PORTFOLIO_SERIES.index.min()
MAX_DATE = PORTFOLIO_SERIES.index.max()
DEFAULT_END = MAX_DATE
DEFAULT_START = max(MIN_DATE, DEFAULT_END - pd.DateOffset(years=3))

BENCHMARK_OPTIONS = [
    {"label": "None", "value": "NONE"},
    {"label": "SPY", "value": "SPY"},
    {"label": "QQQ", "value": "QQQ"},
]
FREQ_OPTIONS = ["Daily", "Weekly"]

CARD_STYLE = {
    "border": "1px solid #e0e0e0",
    "borderRadius": "10px",
    "padding": "12px 14px",
    "boxShadow": "0 1px 2px rgba(0, 0, 0, 0.08)",
    "backgroundColor": "#ffffff",
}
GRID_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
    "gap": "16px",
}
INFO_STYLE = {"cursor": "pointer", "textDecoration": "underline"}
DEBUG_DRAWDOWN_CHECK = False


def _format_pct(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value * 100:.2f}%"


layout = html.Div([
    html.Br(),
    html.H2("Risk & Drawdowns"),
    html.Div(
        risk_free_warning(),
        style={"color": "#b45309", "marginBottom": "8px"},
    ) if risk_free_warning() else html.Div(),
    html.P(
        "Assess portfolio risk, drawdown behavior, and tail-risk measures across the selected window."
    ),

    html.Div([
        html.Div([
            html.Label("Date range"),
            dcc.DatePickerRange(
                id="risk-date-range",
                min_date_allowed=MIN_DATE,
                max_date_allowed=MAX_DATE,
                start_date=DEFAULT_START.date(),
                end_date=DEFAULT_END.date(),
            ),
        ], style={"maxWidth": "360px"}),
        html.Div([
            html.Label("Frequency"),
            dcc.RadioItems(
                id="risk-frequency",
                options=[{"label": f, "value": f} for f in FREQ_OPTIONS],
                value="Daily",
                inline=True,
            ),
        ], style={"maxWidth": "260px"}),
        html.Div([
            html.Label("Benchmark"),
            dcc.Dropdown(
                id="risk-benchmark",
                options=BENCHMARK_OPTIONS,
                value="NONE",
                clearable=False,
            ),
        ], style={"maxWidth": "220px"}),
    ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap"}),

    html.Br(),
    html.H3("Top Risk Metrics"),
    html.Div([
        html.Div([
            html.Div(id="risk-vol-metric", style={"fontSize": "20px", "fontWeight": "600"}),
            html.Small("Annualized volatility"),
        ], style=CARD_STYLE),
        html.Div([
            html.Div(id="risk-dd-metric", style={"fontSize": "20px", "fontWeight": "600"}),
            html.Small("Max drawdown"),
        ], style=CARD_STYLE),
        html.Div([
            html.Div(id="risk-sharpe-metric", style={"fontSize": "20px", "fontWeight": "600"}),
            html.Small("Sharpe ratio (Rf=0)"),
        ], style=CARD_STYLE),
        html.Div([
            html.Div(id="risk-calmar-metric", style={"fontSize": "20px", "fontWeight": "600"}),
            html.Small("Calmar ratio (CAGR / max drawdown)"),
        ], style=CARD_STYLE),
    ], style=GRID_STYLE),

    html.Br(),
    html.H3("Underwater Drawdown"),
    html.Div([
        html.Span("Drawdown plot", id="underwater-info", style=INFO_STYLE),
        dbc.Tooltip(
            "Drawdown is percent decline from the running peak. Shows depth of losses and recovery.",
            target="underwater-info",
            placement="right",
        ),
    ]),
    dcc.Loading(dcc.Graph(id="underwater-graph")),

    html.Br(),
    html.H3("Drawdown Episodes"),
    html.P("Worst 10 drawdown episodes with depth, duration to trough, and recovery."),
    dash_table.DataTable(
        id="drawdown-table",
        columns=[
            {"name": "Peak", "id": "peak"},
            {"name": "Trough", "id": "trough"},
            {"name": "Recovery", "id": "recovery"},
            {"name": "Depth", "id": "depth"},
            {"name": "Periods to Trough", "id": "periods_to_trough"},
            {"name": "Periods to Recovery", "id": "periods_to_recovery"},
            {"name": "Total Periods", "id": "total_periods"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px"},
    ),
    dcc.Loading(dcc.Graph(id="drawdown-scatter")),

    html.Br(),
    html.H3("VaR / CVaR (Historical)"),
    html.Div([
        html.Span("What do VaR/CVaR mean?", id="var-info", style=INFO_STYLE),
        dbc.Tooltip(
            "VaR is the loss threshold exceeded only alpha% of the time; CVaR is the average loss beyond VaR.",
            target="var-info",
            placement="right",
        ),
    ]),
    dcc.Loading(dcc.Graph(id="var-rolling-graph")),
    html.Div(id="var-summary"),
])


@callback(
    Output("risk-vol-metric", "children"),
    Output("risk-dd-metric", "children"),
    Output("risk-sharpe-metric", "children"),
    Output("risk-calmar-metric", "children"),
    Output("underwater-graph", "figure"),
    Output("drawdown-table", "data"),
    Output("drawdown-scatter", "figure"),
    Output("var-rolling-graph", "figure"),
    Output("var-summary", "children"),
    Input("risk-date-range", "start_date"),
    Input("risk-date-range", "end_date"),
    Input("risk-frequency", "value"),
    Input("risk-benchmark", "value"),
)
def update_risk_dashboard(start_date, end_date, freq, benchmark):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    prices = PORTFOLIO_SERIES.loc[start:end].dropna()
    if prices.empty:
        empty = empty_figure("No data available for selected range.")
        return (
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            empty,
            [],
            empty,
            empty,
            "No data available.",
        )

    benchmark_series = None
    if benchmark and benchmark != "NONE":
        bench_prices = load_ticker_prices([benchmark], start=start, end=end)
        if not bench_prices.empty and benchmark in bench_prices.columns:
            benchmark_series = bench_prices[benchmark].dropna()

    prices_plot = prices
    if freq == "Weekly":
        prices_plot = prices.resample("W-FRI").last()
        if benchmark_series is not None:
            benchmark_series = benchmark_series.resample("W-FRI").last()

    port_ret = compute_returns(prices, freq)
    periods = 252 if freq == "Daily" else 52

    vol = annualized_vol(port_ret, periods_per_year=periods)
    dd = max_drawdown(prices_plot)
    cagr = np.nan
    if len(prices_plot) > 1:
        days = (prices_plot.index[-1] - prices_plot.index[0]).days
        years = days / 365.25 if days > 0 else np.nan
        if years and years > 0:
            cagr = (prices_plot.iloc[-1] / prices_plot.iloc[0]) ** (1 / years) - 1
    sharpe = (port_ret.mean() * periods) / vol if vol and vol > 0 else np.nan
    calmar = cagr / abs(dd) if dd and dd < 0 else np.nan

    dd_series = drawdown_series(prices_plot)
    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(x=dd_series.index, y=dd_series.values, mode="lines", name="Portfolio"))
    if benchmark_series is not None and not benchmark_series.empty:
        bench_dd = drawdown_series(benchmark_series)
        dd_fig.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd.values, mode="lines", name=benchmark))
    dd_fig.update_layout(title="Underwater Drawdown", height=450, legend_title_text="")
    dd_fig.update_yaxes(title_text="Drawdown", tickformat=".1%")

    episodes = drawdown_episodes(prices_plot)
    episodes = episodes[:10]
    if DEBUG_DRAWDOWN_CHECK:
        dd_min = float(dd_series.min()) if not dd_series.empty else np.nan
        ep_min = min((ep["depth"] for ep in episodes), default=np.nan)
        if pd.notna(dd_min) and pd.notna(ep_min) and not np.isclose(dd_min, ep_min, atol=1e-6):
            print(f"[WARN] Drawdown mismatch: series {dd_min:.6f} vs episodes {ep_min:.6f}")
    rows = []
    scatter_rows = []
    for ep in episodes:
        recovery = ep["recovery"]
        rows.append({
            "peak": ep["peak"].date().isoformat(),
            "trough": ep["trough"].date().isoformat(),
            "recovery": recovery.date().isoformat() if recovery is not None else "Unrecovered",
            "depth": _format_pct(ep["depth"]),
            "periods_to_trough": ep["duration_to_trough"],
            "periods_to_recovery": ep["recovery_time"] if ep["recovery_time"] is not None else "n/a",
            "total_periods": ep["total_duration"],
        })
        scatter_rows.append({
            "depth": abs(ep["depth"]),
            "duration": ep["duration_to_trough"],
        })

    scatter_fig = empty_figure("No drawdown episodes to display.")
    if scatter_rows:
        df_scatter = pd.DataFrame(scatter_rows)
        scatter_fig = px.scatter(
            df_scatter,
            x="duration",
            y="depth",
            title="Drawdown Depth vs Duration to Trough",
        )
        scatter_fig.update_layout(height=350)
        scatter_fig.update_yaxes(title_text="Depth", tickformat=".1%")
        scatter_fig.update_xaxes(title_text="Periods to Trough")

    var_alpha = 0.05
    var_95, cvar_95 = var_cvar(port_ret, alpha=var_alpha)
    var_99, cvar_99 = var_cvar(port_ret, alpha=0.01)
    window = 63 if freq == "Daily" else 26
    roll_var = rolling_var(port_ret, window=window, alpha=var_alpha)
    var_fig = px.line(roll_var, title=f"Rolling VaR {int((1 - var_alpha) * 100)}% (window={window})")
    var_fig.update_layout(height=350)
    var_fig.update_yaxes(title_text="VaR", tickformat=".2%")
    var_fig.update_xaxes(title_text="Date")

    exceptions = (port_ret < var_95).sum()
    hit_rate = exceptions / len(port_ret) if len(port_ret) else np.nan
    var_summary = html.Div([
        html.P(f"VaR 95%: {_format_pct(var_95)} | CVaR 95%: {_format_pct(cvar_95)}"),
        html.P(f"VaR 99%: {_format_pct(var_99)} | CVaR 99%: {_format_pct(cvar_99)}"),
        html.P(f"Exceptions (VaR 95%): {exceptions} ({hit_rate:.2%} hit rate)"),
    ])

    return (
        _format_pct(vol),
        _format_pct(dd),
        f"{sharpe:.2f}" if pd.notna(sharpe) else "n/a",
        f"{calmar:.2f}" if pd.notna(calmar) else "n/a",
        dd_fig,
        rows,
        scatter_fig,
        var_fig,
        var_summary,
    )
