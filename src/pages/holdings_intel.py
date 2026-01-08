import dash
from dash import html, dcc, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analytics.holdings_intel import (
    concentration_metrics,
    covariance_matrix,
    risk_contributions,
    scenario_impact,
)
from analytics.risk import compute_returns
from analytics_macro import load_ticker_prices
from analytics.constants import DEFAULT_START_DATE_ANALYSIS
from analytics.portfolio import load_holdings_timeseries, load_portfolio_series
from analytics.common import time_varying_weights
from analytics.common import annualize_return_cagr, annualize_vol
from analytics.regimes import compute_regime_features, label_regimes
from viz.plots import empty_figure
from strategies.factor_rotation import RotationParams, backtest_rotation, ETF_UNIVERSE, monthly_returns as rotation_monthly_returns

dash.register_page(__name__, path="/holdings-intel", name="Holdings Intelligence")

PORTFOLIO_SERIES = load_portfolio_series()
MIN_DATE = PORTFOLIO_SERIES.index.min()
MAX_DATE = PORTFOLIO_SERIES.index.max()
DEFAULT_END = MAX_DATE
DEFAULT_START = max(MIN_DATE, DEFAULT_START_DATE_ANALYSIS)

INFO_STYLE = {"cursor": "pointer", "textDecoration": "underline"}


layout = html.Div([
    html.Br(),
    html.H2("Holdings Intelligence"),
    html.P("Analyze concentration, risk contribution, and scenario shocks."),
    html.Br(),
    html.Br(),

    html.Div([
    html.Div([
        html.Label("Date range"),
        dcc.DatePickerRange(
            id="holdings-date-range",
            min_date_allowed=MIN_DATE,
            max_date_allowed=MAX_DATE,
            start_date=DEFAULT_START.date(),
            end_date=DEFAULT_END.date(),
        ),
    ], style={"maxWidth": "360px"}),
    html.Div([
        html.Label("Frequency"),
        dcc.RadioItems(
            id="holdings-frequency",
            options=[{"label": f, "value": f} for f in ["Daily", "Weekly"]],
            value="Daily",
            inline=True,
        ),
    ], style={"maxWidth": "260px"}),
    ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap"}),

    html.Br(),
    html.Br(),
    html.H3("Concentration"),
    html.Br(),
    html.P("HHI and top concentration based on end-of-period weights."),
    html.Div(id="concentration-metrics"),

    html.Br(),
    html.Hr(),
    html.Br(),
    html.H3("Risk Contribution"),
    html.Br(),
    html.P("Top 10 holdings by percent contribution to portfolio risk."),
    html.Div(id="risk-contrib-method", style={"color": "#5b6675", "marginBottom": "6px"}),
    dcc.Loading(dcc.Graph(id="risk-contrib-graph")),
    dash_table.DataTable(
        id="risk-contrib-table",
        columns=[
            {"name": "Holding", "id": "holding"},
            {"name": "Weight", "id": "weight"},
            {"name": "Vol", "id": "vol"},
            {"name": "PCR", "id": "pcr"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px"},
    ),

    html.Br(),
    html.Hr(),
    html.Br(),
    html.H3("Scenario Shocks"),
    html.Br(),
    html.Div([
        html.Label("Holding"),
        dcc.Dropdown(id="shock-holding", options=[], value=None),
    ], style={"maxWidth": "260px"}),
    html.Div([
        html.Label("Shock (%)"),
        dcc.Slider(
            id="shock-slider",
            min=-30,
            max=30,
            step=1,
            value=-5,
            marks={-30: "-30%", -10: "-10%", 0: "0%", 10: "10%", 30: "30%"},
        ),
    ], style={"maxWidth": "420px"}),
    html.Div(id="shock-impact-output"),

    html.Br(),
    html.Hr(),
    html.Br(),
    html.H3("Macro Shock Presets"),
    html.Br(),
    html.P("Preset proxy shocks; portfolio impact uses simple proxy weight assumptions."),
    html.Div([
        html.Span("Rates shock (TLT -5% week)", style=INFO_STYLE),
        html.Span(" | "),
        html.Span("Credit shock (HYG -3% week)", style=INFO_STYLE),
        html.Span(" | "),
        html.Span("Oil shock (USO +8% week)", style=INFO_STYLE),
    ]),
    html.Div(id="macro-shock-output"),

    html.Br(),
    html.Hr(),
    html.Br(),
    html.H3("Factor Rotation Sleeve (Monthly, Blended Weights)"),
    html.Br(),
    html.P("Monthly rebalanced sleeve with blended momentum/trend signals, turnover control, and transaction costs."),
    html.Div([
        html.Div([
            html.Label("Date range"),
            dcc.DatePickerRange(
                id="rotation-date-range",
                min_date_allowed=MIN_DATE,
                max_date_allowed=MAX_DATE,
                start_date=DEFAULT_START.date(),
                end_date=DEFAULT_END.date(),
            ),
        ], style={"maxWidth": "360px"}),
        html.Div([
            html.Label("Smoothing λ"),
            dcc.Slider(
                id="rotation-smooth",
                min=0.1,
                max=1.0,
                step=0.05,
                value=0.35,
                marks={0.1: "0.1", 0.35: "0.35", 0.7: "0.7", 1.0: "1.0"},
            ),
        ], style={"maxWidth": "320px"}),
        html.Div([
            html.Label("Max weight cap"),
            dcc.Slider(
                id="rotation-max-weight",
                min=0.25,
                max=0.6,
                step=0.05,
                value=0.45,
                marks={0.25: "0.25", 0.45: "0.45", 0.6: "0.6"},
            ),
        ], style={"maxWidth": "320px"}),
        html.Div([
            html.Label("Transaction cost (bps)"),
            dcc.Slider(
                id="rotation-tc",
                min=0,
                max=20,
                step=1,
                value=8,
                marks={0: "0", 5: "5", 10: "10", 20: "20"},
            ),
        ], style={"maxWidth": "320px"}),
        html.Div([
            html.Label("Benchmark"),
            dcc.Dropdown(
                id="rotation-benchmark",
                options=[{"label": b, "value": b} for b in ["SPY", "QQQ"]],
                value="SPY",
                clearable=False,
            ),
        ], style={"maxWidth": "220px"}),
        html.Div([
            html.Label("Sleeve allocation"),
            dcc.Slider(
                id="rotation-alloc",
                min=0.0,
                max=0.5,
                step=0.05,
                value=0.2,
                marks={0.0: "0%", 0.2: "20%", 0.4: "40%", 0.5: "50%"},
            ),
        ], style={"maxWidth": "320px"}),
        html.Div([
            html.Label("Notional ($)"),
            dcc.Input(id="rotation-notional", type="number", value=100000, min=10000, step=1000),
        ], style={"maxWidth": "200px"}),
        html.Div([
            html.Label("Options"),
            dcc.Checklist(
                id="rotation-options",
                options=[
                    {"label": "Trend filter", "value": "trend"},
                    {"label": "Vol adjust", "value": "vol"},
                    {"label": "Regime tilt", "value": "regime"},
                ],
                value=["trend", "vol"],
                inline=True,
            ),
        ], style={"maxWidth": "420px"}),
    ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap"}),

    html.Br(),
    dcc.Loading(dcc.Graph(id="rotation-weights")),
    dcc.Loading(dcc.Graph(id="rotation-equity")),
    html.P("Paper overlay comparison: Core portfolio blended with the sleeve at the chosen allocation."),
    dash_table.DataTable(
        id="rotation-summary",
        columns=[
            {"name": "Metric", "id": "metric"},
            {"name": "Value", "id": "value"},
        ],
        style_table={"overflowX": "auto", "maxWidth": "520px"},
        style_cell={"textAlign": "left", "padding": "6px"},
    ),
    html.Br(),
    dcc.Loading(dcc.Graph(id="rotation-attrib")),
    html.Br(),
    html.H4("Trade Blotter (Suggested Monthly Rebalance Trades)"),
    html.Br(),
    html.P("Trades are based on target weights; transaction costs are applied in the sleeve backtest."),
    dbc.Button("Download CSV", id="rotation-download-btn", size="sm", color="secondary"),
    dcc.Download(id="rotation-download"),
    dash_table.DataTable(
        id="rotation-blotter",
        columns=[
            {"name": "Rebalance Date", "id": "date"},
            {"name": "Asset", "id": "asset"},
            {"name": "Prior Weight", "id": "prior_weight"},
            {"name": "Target Weight", "id": "target_weight"},
            {"name": "Trade (Δ)", "id": "trade"},
            {"name": "Side", "id": "side"},
            {"name": "Est. $", "id": "dollar"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px"},
    ),
])


@callback(
    Output("shock-holding", "options"),
    Output("shock-holding", "value"),
    Input("holdings-date-range", "start_date"),
    Input("holdings-date-range", "end_date"),
)
def update_holdings_options(start_date, end_date):
    holdings_ts = load_holdings_timeseries()
    df = holdings_ts.loc[start_date:end_date]
    mv_cols = [c for c in df.columns if c.startswith("MV_")]
    if not mv_cols:
        return [], None
    last_mvs = df[mv_cols].iloc[-1].fillna(0.0)
    weights = last_mvs / last_mvs.sum() if last_mvs.sum() > 0 else last_mvs
    tickers = [c.replace("MV_", "") for c in weights.index]
    options = [{"label": t, "value": t} for t in tickers]
    top = weights.sort_values(ascending=False).index[0].replace("MV_", "") if not weights.empty else None
    return options, top


@callback(
    Output("concentration-metrics", "children"),
    Output("risk-contrib-method", "children"),
    Output("risk-contrib-graph", "figure"),
    Output("risk-contrib-table", "data"),
    Output("shock-impact-output", "children"),
    Output("macro-shock-output", "children"),
    Input("holdings-date-range", "start_date"),
    Input("holdings-date-range", "end_date"),
    Input("holdings-frequency", "value"),
    Input("shock-holding", "value"),
    Input("shock-slider", "value"),
)
def update_holdings_intel(start_date, end_date, freq, shock_holding, shock_value):
    holdings_ts = load_holdings_timeseries()
    df = holdings_ts.loc[start_date:end_date]
    if df.empty:
        empty = empty_figure("No data available for selected range.")
        return "No data available.", None, empty, [], "No data available.", "No data available."

    mv_cols = [c for c in df.columns if c.startswith("MV_")]
    if not mv_cols:
        empty = empty_figure("No holdings available.")
        return "No holdings available.", None, empty, [], "No holdings available.", "No holdings available."

    last_mvs = df[mv_cols].iloc[-1].fillna(0.0)
    total_mv = float(last_mvs.sum())
    weights = last_mvs / total_mv if total_mv > 0 else last_mvs
    weights.index = [c.replace("MV_", "") for c in weights.index]

    conc = concentration_metrics(weights)
    conc_text = html.Div([
        html.P(f"HHI: {conc['hhi']:.3f}"),
        html.P(f"Top 1 weight: {conc['top1']:.1%}"),
        html.P(f"Top 3 weight: {conc['top3']:.1%}"),
    ])

    price_hist = pd.read_csv("data/historical_prices.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    price_hist = price_hist.loc[start_date:end_date]
    price_hist = price_hist[price_hist.columns.intersection(weights.index)]
    if price_hist.empty:
        risk_fig = empty_figure("No price history for risk contribution.")
        return conc_text, None, risk_fig, [], "No price data for scenario.", "No price data for macro shocks."

    returns = compute_returns(price_hist, freq=freq)
    if returns.empty:
        risk_fig = empty_figure("Not enough data for risk contribution.")
        return conc_text, None, risk_fig, [], "No data for scenario.", "No data for macro shocks."

    weights_lag = time_varying_weights(df[mv_cols], freq=freq)
    method_note = "Method: time-varying weights (lagged MV weights) across the window."
    avg_weights = weights_lag.reindex(returns.index, method="ffill").mean()
    if weights_lag.empty or float(avg_weights.sum()) == 0.0:
        method_note = "Method: snapshot weights (fallback; limited history in window)."
        avg_weights = weights.copy()
    cov = covariance_matrix(returns)
    risk_df = risk_contributions(avg_weights, cov)
    if risk_df.empty:
        risk_fig = empty_figure("Risk contribution unavailable.")
        return conc_text, method_note, risk_fig, [], "No data for scenario.", "No data for macro shocks."

    risk_df["holding"] = risk_df.index
    risk_df["vol"] = np.sqrt(np.diag(cov))
    top_risk = risk_df.sort_values("pcr", ascending=False).head(10)
    risk_fig = px.bar(
        top_risk.sort_values("pcr"),
        x="pcr",
        y="holding",
        orientation="h",
        title="Top Risk Contributors",
    )
    risk_fig.update_layout(height=450, showlegend=False)
    risk_fig.update_xaxes(title_text="% Risk Contribution", tickformat=".1%")

    table_rows = []
    for _, row in top_risk.iterrows():
        table_rows.append({
            "holding": row["holding"],
            "weight": f"{row['weight']:.1%}",
            "vol": f"{row['vol']:.2%}",
            "pcr": f"{row['pcr']:.1%}",
        })

    shock_val = (shock_value or 0) / 100.0
    shock_text = "Select a holding for scenario."
    if shock_holding and shock_holding in weights.index:
        impact = scenario_impact(weights, {shock_holding: shock_val})
        shock_text = f"Estimated immediate impact: {impact:.2%} (weight {weights[shock_holding]:.1%} × shock {shock_val:.1%})"

    macro_msg = "Macro shocks shown as proxy scenarios (beta-based integration pending factor model)."

    return conc_text, method_note, risk_fig, table_rows, shock_text, macro_msg


@callback(
    Output("rotation-weights", "figure"),
    Output("rotation-equity", "figure"),
    Output("rotation-summary", "data"),
    Output("rotation-attrib", "figure"),
    Output("rotation-blotter", "data"),
    Input("rotation-date-range", "start_date"),
    Input("rotation-date-range", "end_date"),
    Input("rotation-smooth", "value"),
    Input("rotation-max-weight", "value"),
    Input("rotation-tc", "value"),
    Input("rotation-benchmark", "value"),
    Input("rotation-alloc", "value"),
    Input("rotation-options", "value"),
    Input("rotation-notional", "value"),
)
def update_factor_rotation(start_date, end_date, smooth_lambda, max_weight, tc_bps, benchmark, alloc, options, notional):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    options = options or []
    notional = float(notional or 100000)

    prices = load_ticker_prices(ETF_UNIVERSE, start=start, end=end)
    if prices.empty:
        empty = empty_figure("No data available for selected range.")
        return empty, empty, [], empty, []

    regime_labels = None
    if "regime" in options:
        features = compute_regime_features(prices, freq="Daily")
        regime_labels = label_regimes(features, "Balanced")

    params = RotationParams(
        smooth_lambda=float(smooth_lambda or 0.35),
        max_weight=float(max_weight or 0.45),
        tc_bps=float(tc_bps or 0.0),
        vol_adjust="vol" in options,
        trend_filter="trend" in options,
        regime_tilt="regime" in options,
    )

    results = backtest_rotation(prices, params, regime_labels)
    if not results:
        empty = empty_figure("Not enough data to compute signals.")
        return empty, empty, [], empty, []

    weights = results["weights"]
    returns = results["returns"]
    equity = results["equity"]
    turnover = results["turnover"]
    monthly_rets = results["monthly_returns"]

    weights_fig = empty_figure("No weights to display.")
    if not weights.empty:
        weights_fig = px.area(weights, title="Target Weights Over Time")
        weights_fig.update_layout(height=420, legend_title_text="")
        weights_fig.update_yaxes(tickformat=".0%")

    bench_prices = load_ticker_prices([benchmark], start=weights.index.min(), end=weights.index.max())
    bench_ret = rotation_monthly_returns(bench_prices[[benchmark]]) if not bench_prices.empty else pd.DataFrame()
    bench_ret = bench_ret[benchmark].reindex(returns.index).fillna(0.0) if not bench_ret.empty else pd.Series(dtype=float)

    core = load_portfolio_series().loc[start:end]
    core_m = core.resample("M").last().pct_change().replace([np.inf, -np.inf], np.nan)
    core_m = core_m.reindex(returns.index).fillna(0.0)
    blended = (1 - float(alloc or 0.0)) * core_m + float(alloc or 0.0) * returns

    equity_fig = go.Figure()
    equity_fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Rotation Sleeve"))
    if not bench_ret.empty:
        equity_fig.add_trace(go.Scatter(x=bench_ret.index, y=(1 + bench_ret).cumprod(), name=benchmark))
    equity_fig.add_trace(go.Scatter(x=blended.index, y=(1 + blended).cumprod(), name="Blended Core+Sleeve"))
    equity_fig.update_layout(title="Equity Curve (Monthly)", height=420, legend_title_text="")

    ann_return = annualize_return_cagr(returns, 12)
    ann_vol = annualize_vol(returns, 12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    dd = (equity / equity.cummax() - 1).min()
    calmar = ann_return / abs(dd) if dd < 0 else np.nan
    te = (returns - bench_ret).std() * np.sqrt(12) if not bench_ret.empty else np.nan
    ir = (returns - bench_ret).mean() / (returns - bench_ret).std() * np.sqrt(12) if not bench_ret.empty else np.nan

    summary = [
        {"metric": "CAGR", "value": f"{ann_return:.2%}"},
        {"metric": "Ann Vol", "value": f"{ann_vol:.2%}"},
        {"metric": "Sharpe (Rf=0)", "value": f"{sharpe:.2f}" if pd.notna(sharpe) else "n/a"},
        {"metric": "Max DD", "value": f"{dd:.2%}"},
        {"metric": "Calmar", "value": f"{calmar:.2f}" if pd.notna(calmar) else "n/a"},
        {"metric": "Avg Monthly Turnover", "value": f"{turnover.mean():.2%}"},
        {"metric": f"Tracking Error vs {benchmark}", "value": f"{te:.2%}" if pd.notna(te) else "n/a"},
        {"metric": f"Info Ratio vs {benchmark}", "value": f"{ir:.2f}" if pd.notna(ir) else "n/a"},
    ]

    contrib = (results["weights_lag"] * monthly_rets).reindex(returns.index)
    contrib_fig = empty_figure("No contribution data.")
    if not contrib.empty:
        contrib_fig = px.bar(
            contrib,
            title="Monthly Contribution by Asset",
            barmode="relative",
        )
        contrib_fig.update_layout(height=420, legend_title_text="")
        contrib_fig.update_yaxes(tickformat=".1%")

    blotter_rows = []
    weights_diff = weights.diff().fillna(weights.iloc[0])
    for dt, row in weights_diff.iterrows():
        for asset, delta in row.items():
            prior = weights.loc[dt, asset] - delta
            side = "Buy" if delta > 0 else "Sell" if delta < 0 else "Hold"
            blotter_rows.append({
                "date": dt.date().isoformat(),
                "asset": asset,
                "prior_weight": f"{prior:.2%}",
                "target_weight": f"{weights.loc[dt, asset]:.2%}",
                "trade": f"{delta:.2%}",
                "side": side,
                "dollar": f"{delta * notional:,.0f}",
            })

    return weights_fig, equity_fig, summary, contrib_fig, blotter_rows


@callback(
    Output("rotation-download", "data"),
    Input("rotation-download-btn", "n_clicks"),
    State("rotation-blotter", "data"),
    prevent_initial_call=True,
)
def download_rotation_blotter(n_clicks, rows):
    if not n_clicks or not rows:
        return None
    df = pd.DataFrame(rows)
    return dcc.send_data_frame(df.to_csv, "rotation_trade_blotter.csv", index=False)
