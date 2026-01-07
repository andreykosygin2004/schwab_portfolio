import dash
from dash import html, dcc, Input, Output, callback, dash_table, ctx
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
    returns_from_prices,
    simulate_overlay,
    transition_matrix,
)
from analytics.overlay import (
    apply_overlay,
    compute_overlay_weights,
    overlay_summary_stats,
    rolling_beta,
)
from analytics.risk import drawdown_series, max_drawdown
from analytics.common import annualize_return_cagr, annualize_vol
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


def _perf_summary(returns: pd.Series, periods: int) -> dict:
    if returns.empty:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "max_dd": np.nan,
            "sharpe": np.nan,
            "win_rate": np.nan,
            "avg_return": np.nan,
        }
    ann_return = annualize_return_cagr(returns, periods)
    ann_vol = annualize_vol(returns, periods)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    win_rate = (returns > 0).mean()
    max_dd = max_drawdown((1 + returns).cumprod())
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "avg_return": returns.mean(),
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
    dcc.Loading(dcc.Graph(id="regime-timeline")),

    html.Br(),
    html.H3("Regime Summary"),
    dash_table.DataTable(
        id="regime-summary-table",
        columns=[
            {"name": "Regime", "id": "regime"},
            {"name": "Days", "id": "days"},
            {"name": "Avg Return", "id": "avg_return"},
            {"name": "Ann Return", "id": "ann_return"},
            {"name": "Ann Vol", "id": "ann_vol"},
            {"name": "Max DD", "id": "max_dd"},
            {"name": "Sharpe", "id": "sharpe"},
            {"name": "Win Rate", "id": "win_rate"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px"},
    ),
    html.Div(id="regime-warning", style={"color": "#b45309", "marginTop": "6px"}),

    html.Br(),
    html.H3("Transition Matrix"),
    dcc.Loading(dcc.Graph(id="regime-transition")),

    html.Br(),
    html.H3("Regime Return Distribution (Same Frequency)"),
    dcc.Loading(dcc.Graph(id="regime-dist")),

    html.Br(),
    html.H3("Overlay Simulator"),
    html.P("Overlay is simulated only. Charts below compare baseline vs overlay and benchmark-relative metrics."),
    html.Div([
        html.Label("Overlay Preset"),
        dcc.Dropdown(
            id="overlay-preset",
            options=[{"label": p, "value": p} for p in PRESETS],
            value="Balanced",
            clearable=False,
        ),
    ], style={"maxWidth": "240px"}),
    html.Div([
        html.Label("Allow leverage"),
        dcc.Checklist(
            id="overlay-leverage",
            options=[{"label": "Allow leverage", "value": "on"}],
            value=[],
            inline=True,
        ),
    ]),
    html.Div([
        html.Label("Overlay Settings (editable)"),
        dash_table.DataTable(
            id="overlay-settings-table",
            columns=[
                {"name": "Regime", "id": "regime", "editable": False},
                {"name": "Target Beta Mult", "id": "beta_mult", "type": "numeric"},
                {"name": "TLT Tilt", "id": "tlt_tilt", "type": "numeric"},
                {"name": "GLD Tilt", "id": "gld_tilt", "type": "numeric"},
            ],
            editable=True,
            style_table={"overflowX": "auto", "marginBottom": "12px"},
            style_cell={"textAlign": "left", "padding": "6px"},
        ),
    ]),
    html.Div([
        html.Div(id="overlay-exposure-summary", style={"fontWeight": "600"}),
    ]),
    dash_table.DataTable(
        id="overlay-mapping-table",
        columns=[
            {"name": "Regime", "id": "regime"},
            {"name": "Multiplier", "id": "multiplier"},
            {"name": "Pct of periods", "id": "pct_time"},
        ],
        style_table={"overflowX": "auto", "marginBottom": "12px"},
        style_cell={"textAlign": "left", "padding": "6px"},
    ),
    dcc.Loading(dcc.Graph(id="overlay-cumret")),
    dcc.Loading(dcc.Graph(id="overlay-drawdown")),
    dcc.Loading(dcc.Graph(id="overlay-weights")),
    dcc.Loading(dcc.Graph(id="overlay-beta")),
    dcc.Loading(dcc.Graph(id="overlay-vol")),
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
    Output("regime-warning", "children"),
    Output("regime-transition", "figure"),
    Output("regime-dist", "figure"),
    Output("overlay-cumret", "figure"),
    Output("overlay-drawdown", "figure"),
    Output("overlay-weights", "figure"),
    Output("overlay-beta", "figure"),
    Output("overlay-vol", "figure"),
    Output("overlay-te", "figure"),
    Output("overlay-summary-table", "data"),
    Output("overlay-exposure-summary", "children"),
    Output("overlay-mapping-table", "data"),
    Output("overlay-settings-table", "data"),
    Input("regime-date-range", "start_date"),
    Input("regime-date-range", "end_date"),
    Input("regime-frequency", "value"),
    Input("regime-benchmark", "value"),
    Input("regime-preset", "value"),
    Input("overlay-preset", "value"),
    Input("overlay-leverage", "value"),
    Input("overlay-settings-table", "data"),
)
def update_regimes(start_date, end_date, freq, benchmark, preset, overlay_preset, overlay_leverage, overlay_table):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    debug = False

    portfolio = PORTFOLIO_SERIES.loc[start:end].dropna()
    if portfolio.empty:
        empty = empty_figure("No data available.")
        return empty, [], "", empty, empty, empty, empty, empty, empty, empty, [], "", [], []

    prices = load_proxy_prices(start, end)
    if prices.empty:
        empty = empty_figure("No proxy data available.")
        return empty, [], "", empty, empty, empty, empty, empty, empty, empty, [], "", [], []

    features = compute_regime_features(prices, freq)
    labels = label_regimes(features, preset)
    labels = labels.reindex(features.index).dropna()
    if labels.empty:
        empty = empty_figure("No regime labels available.")
        return empty, [], "", empty, empty, empty, empty, empty, empty, empty, [], "", [], []

    equity = portfolio.reindex(labels.index).ffill().dropna()
    if equity.empty:
        empty = empty_figure("No aligned portfolio data.")
        return empty, [], "", empty, empty, empty, empty, empty, empty, empty, [], "", [], []

    periods = 252 if freq == "Daily" else 52
    port_ret = returns_from_prices(equity, freq=freq)
    labels = labels.reindex(port_ret.index).dropna()
    port_ret = port_ret.reindex(labels.index).dropna()

    if debug:
        print("[Regimes] window", start.date(), end.date(), "freq", freq)
        print("[Regimes] obs", len(port_ret), "labels", labels.value_counts().to_dict())
        print("[Regimes] ann factor", periods)
        print("[Regimes] port_ret head", port_ret.head().to_dict())
        print("[Regimes] port_ret tail", port_ret.tail().to_dict())
        print("[Regimes] NaNs", port_ret.isna().sum())

    timeline_fig = go.Figure()
    equity_plot = equity.reindex(labels.index).ffill()
    timeline_fig.add_trace(go.Scatter(x=equity_plot.index, y=equity_plot.values, mode="lines", name="Portfolio"))
    for regime in labels.unique():
        mask = labels == regime
        timeline_fig.add_trace(go.Scatter(
            x=labels.index[mask],
            y=[equity_plot.max() * 0.98] * mask.sum(),
            mode="markers",
            name=regime,
            marker={"size": 6},
        ))
    timeline_fig.update_layout(title="Regime Timeline (Portfolio + Regime Markers)", height=450, legend_title_text="")

    summary_rows = []
    for regime, r in port_ret.groupby(labels):
        stats = _perf_summary(r, periods)
        summary_rows.append({
            "regime": regime,
            "days": len(r),
            "avg_return": _format_pct(stats["avg_return"]),
            "ann_return": _format_pct(stats["ann_return"]),
            "ann_vol": _format_pct(stats["ann_vol"]),
            "max_dd": _format_pct(stats["max_dd"]),
            "sharpe": f"{stats['sharpe']:.2f}" if pd.notna(stats["sharpe"]) else "n/a",
            "win_rate": _format_pct(stats["win_rate"]),
        })
    regime_warning = ""
    if len(summary_rows) < 3:
        counts = labels.value_counts().to_dict()
        regime_warning = f"Only {len(summary_rows)} regimes in this window. Counts: {counts}"

    trans = transition_matrix(labels)
    transition_fig = empty_figure("No transition data.")
    if not trans.empty:
        transition_fig = px.imshow(trans, text_auto=".2f", title="Regime Transition Matrix")
        transition_fig.update_layout(height=400)

    dist_fig = empty_figure("No return distribution.")
    if not port_ret.empty:
        dist_df = pd.DataFrame({"return": port_ret, "regime": labels.loc[port_ret.index]})
        dist_fig = px.box(
            dist_df,
            x="regime",
            y="return",
            title="Return Distribution by Regime",
        )
        dist_fig.update_layout(height=400)
        dist_fig.update_yaxes(tickformat=".1%")

    overlay_defaults = {
        "Conservative": {
            "Risk-On": 1.0,
            "Neutral / Transition": 0.6,
            "Rates Shock": 0.4,
            "Inflation Shock": 0.5,
            "Risk-Off / Credit Stress": 0.2,
        },
        "Balanced": {
            "Risk-On": 1.0,
            "Neutral / Transition": 0.7,
            "Rates Shock": 0.5,
            "Inflation Shock": 0.6,
            "Risk-Off / Credit Stress": 0.3,
        },
        "Aggressive": {
            "Risk-On": 1.0,
            "Neutral / Transition": 0.8,
            "Rates Shock": 0.6,
            "Inflation Shock": 0.7,
            "Risk-Off / Credit Stress": 0.4,
        },
    }
    regime_order = [
        "Risk-On",
        "Neutral / Transition",
        "Rates Shock",
        "Inflation Shock",
        "Risk-Off / Credit Stress",
    ]
    if ctx.triggered_id == "overlay-preset" or not overlay_table:
        overlay_table = [
            {"regime": r, "beta_mult": overlay_defaults[overlay_preset][r], "tlt_tilt": 0.0, "gld_tilt": 0.0}
            for r in regime_order
        ]

    config = {
        "target_beta_mult": {row["regime"]: float(row.get("beta_mult", 1.0)) for row in overlay_table},
        "tlt_tilt": {row["regime"]: float(row.get("tlt_tilt", 0.0)) for row in overlay_table},
        "gld_tilt": {row["regime"]: float(row.get("gld_tilt", 0.0)) for row in overlay_table},
        "max_hedge": 1.0,
        "gross_cap": 1.5,
        "allow_leverage": "on" in (overlay_leverage or []),
    }

    proxy_prices = load_ticker_prices(["QQQ", "TLT", "GLD"], start=start, end=end)
    qqq_ret = returns_from_prices(proxy_prices["QQQ"], freq=freq) if "QQQ" in proxy_prices.columns else pd.Series(dtype=float)
    tlt_ret = returns_from_prices(proxy_prices["TLT"], freq=freq) if "TLT" in proxy_prices.columns else pd.Series(dtype=float)
    gld_ret = returns_from_prices(proxy_prices["GLD"], freq=freq) if "GLD" in proxy_prices.columns else pd.Series(dtype=float)
    beta_window = 63 if freq == "Daily" else 26
    beta_series = rolling_beta(port_ret, qqq_ret, beta_window).ffill()
    weights = compute_overlay_weights(labels, beta_series, config)
    overlay_ret = apply_overlay(port_ret, qqq_ret, tlt_ret, gld_ret, weights)
    exposure = labels.map(lambda x: config["target_beta_mult"].get(x, 1.0)).reindex(port_ret.index).fillna(1.0)
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
        bench_ret = returns_from_prices(bench_prices[benchmark], freq=freq)
        bench_ret = bench_ret.reindex(port_ret.index).dropna()
        overlay_ret = overlay_ret.reindex(bench_ret.index).dropna()
        if len(bench_ret) < (20 if freq == "Daily" else 12):
            te_fig = empty_figure("Insufficient data for rolling window.")
        else:
            window = 63 if freq == "Daily" else 26
            active = overlay_ret - bench_ret
            te = active.rolling(window).std() * np.sqrt(periods)
            ir = active.rolling(window).mean() * periods / active.rolling(window).std()
            te_fig = go.Figure()
            te_fig.add_trace(go.Scatter(x=te.index, y=te.values, name="Tracking Error"))
            te_fig.add_trace(go.Scatter(x=ir.index, y=ir.values, name="Information Ratio", yaxis="y2"))
            te_fig.update_layout(
                title="Overlay Tracking Error & Information Ratio",
                height=420,
                yaxis=dict(title="Tracking Error", tickformat=".1%"),
                yaxis2=dict(title="Info Ratio", overlaying="y", side="right"),
            )

    base_stats = _perf_summary(port_ret, periods)
    overlay_stats = _perf_summary(overlay_ret, periods)
    overlay_rows = [
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

    periods_label = "days" if freq == "Daily" else "weeks"
    avg_exposure = exposure.mean() if not exposure.empty else np.nan
    exposure_summary = f"Average exposure: {avg_exposure:.2f}x"
    if len(exposure) < (30 if freq == "Daily" else 12):
        exposure_summary += f" (sample size {len(exposure)} {periods_label})"

    counts = labels.value_counts().reindex(regime_order).fillna(0.0)
    total = counts.sum()
    mapping_rows = []
    for regime in regime_order:
        pct = counts.get(regime, 0.0) / total if total > 0 else 0.0
        mapping_rows.append({
            "regime": regime,
            "multiplier": f"{config['target_beta_mult'].get(regime, 1.0):.2f}x",
            "pct_time": f"{pct:.1%}",
        })

    weights_fig = empty_figure("No overlay weights.")
    if not weights.empty:
        weights_fig = px.line(weights, title="Overlay Weights Over Time")
        weights_fig.update_layout(height=420, legend_title_text="")

    beta_fig = empty_figure("No beta data.")
    overlay_beta = rolling_beta(overlay_ret, qqq_ret, beta_window)
    if not overlay_beta.empty:
        beta_fig = go.Figure()
        beta_fig.add_trace(go.Scatter(x=beta_series.index, y=beta_series.values, name="Baseline Beta"))
        beta_fig.add_trace(go.Scatter(x=overlay_beta.index, y=overlay_beta.values, name="Overlay Beta"))
        beta_fig.update_layout(title="Rolling Beta to QQQ", height=420, legend_title_text="")

    vol_fig = empty_figure("No volatility data.")
    if not port_ret.empty:
        roll_window = 63 if freq == "Daily" else 26
        base_vol = port_ret.rolling(roll_window).std() * np.sqrt(periods)
        overlay_vol = overlay_ret.rolling(roll_window).std() * np.sqrt(periods)
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Scatter(x=base_vol.index, y=base_vol.values, name="Baseline Vol"))
        vol_fig.add_trace(go.Scatter(x=overlay_vol.index, y=overlay_vol.values, name="Overlay Vol"))
        vol_fig.update_layout(title="Rolling Volatility", height=420, legend_title_text="")
        vol_fig.update_yaxes(tickformat=".1%")

    diag = overlay_summary_stats(port_ret, overlay_ret, weights, qqq_ret, periods, beta_window)
    if diag:
        overlay_rows.append({
            "series": "Overlay Diagnostics",
            "ann_return": f"Avg hedge {diag['avg_hedge']:.2f}",
            "ann_vol": f"Max hedge {diag['max_hedge']:.2f}",
            "max_dd": f"% hedged {diag['pct_hedged']:.1%}",
            "sharpe": f"Avg beta {diag['avg_beta']:.2f}",
        })

    return (
        timeline_fig,
        summary_rows,
        regime_warning,
        transition_fig,
        dist_fig,
        overlay_cum_fig,
        overlay_dd_fig,
        weights_fig,
        beta_fig,
        vol_fig,
        te_fig,
        overlay_rows,
        exposure_summary,
        mapping_rows,
        overlay_table,
    )
