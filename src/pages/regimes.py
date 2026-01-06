import dash
from dash import html, dcc, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analytics.constants import ANALYSIS_END, ANALYSIS_START
from analytics.portfolio import load_portfolio_series, risk_free_warning
from analytics.regimes import (
    analysis_window,
    compute_regime_features,
    label_regimes,
    load_proxy_prices,
    simulate_overlay,
    transition_matrix,
)
from analytics.risk import compute_returns, drawdown_series, max_drawdown
from analytics_macro import load_ticker_prices
from viz.plots import empty_figure

dash.register_page(__name__, path="/regimes", name="Regimes")

PORTFOLIO_SERIES = load_portfolio_series()

FREQ_OPTIONS = ["Daily", "Weekly"]
BENCHMARK_OPTIONS = ["SPY", "QQQ"]
PRESETS = ["Conservative", "Balanced", "Aggressive"]

DEFAULT_START = ANALYSIS_START
DEFAULT_END = ANALYSIS_END

INFO_STYLE = {"cursor": "pointer", "textDecoration": "underline"}


def _format_pct(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value * 100:.2f}%"


def _perf_summary(returns: pd.Series) -> dict:
    if returns.empty:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "max_dd": np.nan,
            "sharpe": np.nan,
            "win_rate": np.nan,
        }
    periods = 252
    ann_return = (1 + returns).prod() ** (periods / len(returns)) - 1 if len(returns) > 0 else np.nan
    ann_vol = returns.std() * np.sqrt(periods)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    win_rate = (returns > 0).mean()
    max_dd = max_drawdown((1 + returns).cumprod())
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate,
    }


layout = html.Div([
    html.Br(),
    html.H2("Regimes"),
    html.Div(
        risk_free_warning(),
        style={"color": "#b45309", "marginBottom": "8px"},
    ) if risk_free_warning() else html.Div(),
    html.P("Rule-based regime detection using macro proxies and risk signals."),

    html.Div([
        html.Div([
            html.Label("Date range"),
            dcc.DatePickerRange(
                id="regime-date-range",
                min_date_allowed=PORTFOLIO_SERIES.index.min(),
                max_date_allowed=PORTFOLIO_SERIES.index.max(),
                start_date=DEFAULT_START.date(),
                end_date=DEFAULT_END.date(),
            ),
        ], style={"maxWidth": "360px"}),
        html.Div([
            html.Label("Frequency"),
            dcc.RadioItems(
                id="regime-frequency",
                options=[{"label": f, "value": f} for f in FREQ_OPTIONS],
                value="Daily",
                inline=True,
            ),
        ], style={"maxWidth": "260px"}),
        html.Div([
            html.Label("Benchmark"),
            dcc.Dropdown(
                id="regime-benchmark",
                options=[{"label": b, "value": b} for b in BENCHMARK_OPTIONS],
                value="SPY",
                clearable=False,
            ),
        ], style={"maxWidth": "220px"}),
        html.Div([
            html.Label("Sensitivity"),
            dcc.Dropdown(
                id="regime-preset",
                options=[{"label": p, "value": p} for p in PRESETS],
                value="Balanced",
                clearable=False,
            ),
        ], style={"maxWidth": "220px"}),
    ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap"}),

    html.Br(),
    html.H3("Regime Timeline"),
    dcc.Loading(dcc.Graph(id="regime-timeline")),

    html.Br(),
    html.H3("Regime Summary"),
    dash_table.DataTable(
        id="regime-summary-table",
        columns=[
            {"name": "Regime", "id": "regime"},
            {"name": "Days", "id": "days"},
            {"name": "Ann Return", "id": "ann_return"},
            {"name": "Ann Vol", "id": "ann_vol"},
            {"name": "Max DD", "id": "max_dd"},
            {"name": "Sharpe", "id": "sharpe"},
            {"name": "Win Rate", "id": "win_rate"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px"},
    ),

    html.Br(),
    html.H3("Transition Matrix"),
    dcc.Loading(dcc.Graph(id="regime-transition")),

    html.Br(),
    html.H3("Regime Return Distribution"),
    dcc.Loading(dcc.Graph(id="regime-dist")),

    html.Br(),
    html.H3("Overlay Simulator"),
    html.P("Exposure multiplier is applied to portfolio returns by regime."),
    dcc.Loading(dcc.Graph(id="overlay-cumret")),
    dcc.Loading(dcc.Graph(id="overlay-drawdown")),
    dcc.Loading(dcc.Graph(id="overlay-te")),
    dash_table.DataTable(
        id="overlay-summary-table",
        columns=[
            {"name": "Series", "id": "series"},
            {"name": "Ann Return", "id": "ann_return"},
            {"name": "Ann Vol", "id": "ann_vol"},
            {"name": "Max DD", "id": "max_dd"},
            {"name": "Sharpe", "id": "sharpe"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px"},
    ),
])


@callback(
    Output("regime-timeline", "figure"),
    Output("regime-summary-table", "data"),
    Output("regime-transition", "figure"),
    Output("regime-dist", "figure"),
    Output("overlay-cumret", "figure"),
    Output("overlay-drawdown", "figure"),
    Output("overlay-te", "figure"),
    Output("overlay-summary-table", "data"),
    Input("regime-date-range", "start_date"),
    Input("regime-date-range", "end_date"),
    Input("regime-frequency", "value"),
    Input("regime-benchmark", "value"),
    Input("regime-preset", "value"),
)
def update_regimes(start_date, end_date, freq, benchmark, preset):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    portfolio = PORTFOLIO_SERIES.loc[start:end].dropna()
    if portfolio.empty:
        empty = empty_figure("No data available.")
        return empty, [], empty, empty, empty, empty, empty, []

    prices = load_proxy_prices(start, end)
    if prices.empty:
        empty = empty_figure("No proxy data available.")
        return empty, [], empty, empty, empty, empty, empty, []

    features = compute_regime_features(prices, freq)
    labels = label_regimes(features, preset)
    labels = labels.reindex(features.index).dropna()
    if labels.empty:
        empty = empty_figure("No regime labels available.")
        return empty, [], empty, empty, empty, empty, empty, []

    equity = portfolio.reindex(labels.index).ffill().dropna()
    if equity.empty:
        empty = empty_figure("No aligned portfolio data.")
        return empty, [], empty, empty, empty, empty, empty, []

    timeline_fig = go.Figure()
    timeline_fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Portfolio"))
    for regime in labels.unique():
        mask = labels == regime
        timeline_fig.add_trace(go.Scatter(
            x=labels.index[mask],
            y=[equity.max() * 0.98] * mask.sum(),
            mode="markers",
            name=regime,
            marker={"size": 6},
        ))
    timeline_fig.update_layout(title="Regime Timeline (Portfolio + Regime Markers)", height=450, legend_title_text="")

    port_ret = compute_returns(equity, freq="Daily")
    port_ret = port_ret.reindex(labels.index).dropna()
    summary_rows = []
    for regime in labels.unique():
        r = port_ret[labels.loc[port_ret.index] == regime]
        stats = _perf_summary(r)
        summary_rows.append({
            "regime": regime,
            "days": len(r),
            "ann_return": _format_pct(stats["ann_return"]),
            "ann_vol": _format_pct(stats["ann_vol"]),
            "max_dd": _format_pct(stats["max_dd"]),
            "sharpe": f"{stats['sharpe']:.2f}" if pd.notna(stats["sharpe"]) else "n/a",
            "win_rate": _format_pct(stats["win_rate"]),
        })

    trans = transition_matrix(labels)
    transition_fig = empty_figure("No transition data.")
    if not trans.empty:
        transition_fig = px.imshow(trans, text_auto=".2f", title="Regime Transition Matrix")
        transition_fig.update_layout(height=400)

    dist_fig = empty_figure("No return distribution.")
    if not port_ret.empty:
        dist_fig = px.box(
            pd.DataFrame({"return": port_ret, "regime": labels.loc[port_ret.index]}),
            x="regime",
            y="return",
            title="Return Distribution by Regime",
        )
        dist_fig.update_layout(height=400)
        dist_fig.update_yaxes(tickformat=".1%")

    exposure_map = {
        "Risk-On": 1.0,
        "Neutral / Transition": 0.8,
        "Rates Shock": 0.6,
        "Inflation Shock": 0.7,
        "Risk-Off / Credit Stress": 0.4,
    }
    overlay_ret, exposure = simulate_overlay(port_ret, labels, exposure_map)
    base_cum = (1 + port_ret).cumprod() - 1
    overlay_cum = (1 + overlay_ret).cumprod() - 1

    overlay_cum_fig = go.Figure()
    overlay_cum_fig.add_trace(go.Scatter(x=base_cum.index, y=base_cum.values, name="Baseline"))
    overlay_cum_fig.add_trace(go.Scatter(x=overlay_cum.index, y=overlay_cum.values, name="Overlay"))
    overlay_cum_fig.update_layout(title="Cumulative Return: Baseline vs Overlay", height=420, legend_title_text="")
    overlay_cum_fig.update_yaxes(tickformat=".1%")

    base_dd = drawdown_series((1 + port_ret).cumprod())
    overlay_dd = drawdown_series((1 + overlay_ret).cumprod())
    overlay_dd_fig = go.Figure()
    overlay_dd_fig.add_trace(go.Scatter(x=base_dd.index, y=base_dd.values, name="Baseline"))
    overlay_dd_fig.add_trace(go.Scatter(x=overlay_dd.index, y=overlay_dd.values, name="Overlay"))
    overlay_dd_fig.update_layout(title="Drawdown: Baseline vs Overlay", height=420, legend_title_text="")
    overlay_dd_fig.update_yaxes(tickformat=".1%")

    bench_prices = load_ticker_prices([benchmark], start=start, end=end)
    te_fig = empty_figure("No benchmark data.")
    if not bench_prices.empty and benchmark in bench_prices.columns:
        bench_ret = compute_returns(bench_prices[benchmark].reindex(port_ret.index).dropna(), freq="Daily")
        te = (overlay_ret - bench_ret).rolling(63).std() * np.sqrt(252)
        ir = (overlay_ret - bench_ret).rolling(63).mean() * 252 / te
        te_fig = go.Figure()
        te_fig.add_trace(go.Scatter(x=te.index, y=te.values, name="Tracking Error"))
        te_fig.add_trace(go.Scatter(x=ir.index, y=ir.values, name="Information Ratio", yaxis="y2"))
        te_fig.update_layout(
            title="Overlay Tracking Error & Information Ratio",
            height=420,
            yaxis=dict(title="Tracking Error", tickformat=".1%"),
            yaxis2=dict(title="Info Ratio", overlaying="y", side="right"),
        )

    base_stats = _perf_summary(port_ret)
    overlay_stats = _perf_summary(overlay_ret)
    summary_rows = [
        {
            "series": "Baseline",
            "ann_return": _format_pct(base_stats["ann_return"]),
            "ann_vol": _format_pct(base_stats["ann_vol"]),
            "max_dd": _format_pct(base_stats["max_dd"]),
            "sharpe": f"{base_stats['sharpe']:.2f}" if pd.notna(base_stats["sharpe"]) else "n/a",
        },
        {
            "series": "Overlay",
            "ann_return": _format_pct(overlay_stats["ann_return"]),
            "ann_vol": _format_pct(overlay_stats["ann_vol"]),
            "max_dd": _format_pct(overlay_stats["max_dd"]),
            "sharpe": f"{overlay_stats['sharpe']:.2f}" if pd.notna(overlay_stats["sharpe"]) else "n/a",
        },
    ]

    return (
        timeline_fig,
        summary_rows,
        transition_fig,
        dist_fig,
        overlay_cum_fig,
        overlay_dd_fig,
        te_fig,
        summary_rows,
    )
