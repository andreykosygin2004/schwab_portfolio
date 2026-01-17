import dash
from dash import html, dcc, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analytics.constants import DEFAULT_START_DATE_ANALYSIS
from analytics.common import annualize_return_cagr, annualize_vol
from analytics.portfolio import load_portfolio_series
from analytics.regimes import compute_regime_features, label_regimes
from analytics_macro import load_ticker_prices
from viz.plots import empty_figure
from strategies.factor_rotation import (
    RotationParams,
    backtest_rotation,
    ETF_UNIVERSE,
    monthly_returns as rotation_monthly_returns,
)

dash.register_page(__name__, path="/factor-rotation", name="Factor Rotation")

PORTFOLIO_SERIES = load_portfolio_series()
MIN_DATE = PORTFOLIO_SERIES.index.min()
MAX_DATE = PORTFOLIO_SERIES.index.max()
DEFAULT_END = MAX_DATE
DEFAULT_START = max(MIN_DATE, DEFAULT_START_DATE_ANALYSIS)
EXECUTION_DEFAULT = pd.Timestamp.now(tz="America/New_York").normalize().replace(day=1)


layout = html.Div([
    html.Br(),
    html.Br(),
    html.H2("Factor Rotation Sleeve"),
    html.P("Monthly rebalanced sleeve with blended momentum/trend signals, turnover control, and transaction costs."),
    html.Br(),
    html.Br(),

    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Date range"),
                    dcc.DatePickerRange(
                        id="rotation-date-range",
                        min_date_allowed=MIN_DATE,
                        max_date_allowed=MAX_DATE,
                        start_date=DEFAULT_START.date(),
                        end_date=DEFAULT_END.date(),
                    ),
                ], xs=12, md=4),
                dbc.Col([
                    html.Label("Benchmark"),
                    dcc.Dropdown(
                        id="rotation-benchmark",
                        options=[{"label": b, "value": b} for b in ["SPY", "QQQ"]],
                        value="SPY",
                        clearable=False,
                    ),
                ], xs=12, md=3),
                dbc.Col([
                    html.Label("Transaction cost (bps)"),
                    dcc.Slider(
                        id="rotation-tc",
                        min=0,
                        max=20,
                        step=1,
                        value=8,
                        marks={0: "0", 5: "5", 10: "10", 20: "20"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], xs=12, md=5),
            ], className="g-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Smoothing λ"),
                    dcc.Slider(
                        id="rotation-smooth",
                        min=0.1,
                        max=1.0,
                        step=0.05,
                        value=0.35,
                        marks={0.1: "0.10", 0.35: "0.35", 0.7: "0.70", 1.0: "1.00"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], xs=12, md=6),
                dbc.Col([
                    html.Label("Sleeve allocation"),
                    dcc.Slider(
                        id="rotation-alloc",
                        min=0.0,
                        max=0.5,
                        step=0.05,
                        value=0.2,
                        marks={0.0: "0%", 0.25: "25%", 0.5: "50%"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], xs=12, md=6),
            ], className="g-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Max weight cap"),
                    dcc.Slider(
                        id="rotation-max-weight",
                        min=0.25,
                        max=0.6,
                        step=0.05,
                        value=0.45,
                        marks={0.25: "0.25", 0.45: "0.45", 0.6: "0.60"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], xs=12, md=6),
                dbc.Col([
                    html.Label("Notional ($)"),
                    dcc.Input(id="rotation-notional", type="number", value=10000, min=10000, step=1000),
                ], xs=12, md=3),
                dbc.Col([
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
                ], xs=12, md=3),
            ], className="g-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Blotter Mode"),
                    dcc.RadioItems(
                        id="rotation-blotter-mode",
                        options=[
                            {"label": "Backtest timeline", "value": "backtest"},
                            {"label": "Execution timeline", "value": "execution"},
                        ],
                        value="backtest",
                        inline=True,
                    ),
                ], xs=12, md=5),
                dbc.Col([
                    html.Label("Execution Start Date"),
                    dcc.DatePickerSingle(
                        id="rotation-exec-date",
                        min_date_allowed=MIN_DATE,
                        max_date_allowed=MAX_DATE,
                        date=EXECUTION_DEFAULT.date(),
                    ),
                ], xs=12, md=4),
                dbc.Col([
                    html.Label("Current Sleeve Weights"),
                    dcc.Dropdown(
                        id="rotation-current-weights",
                        options=[
                            {"label": "Zero (new sleeve)", "value": "zero"},
                            {"label": "Last target weights", "value": "last"},
                        ],
                        value="zero",
                        clearable=False,
                    ),
                ], xs=12, md=3),
            ], className="g-3"),
        ]),
        style={"borderRadius": "10px", "border": "1px solid #e6e9ee"},
    ),

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
    html.Div(id="rotation-blotter-note", style={"color": "#5b6675", "marginBottom": "6px"}),
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
    html.Br(),
])


@callback(
    Output("rotation-weights", "figure"),
    Output("rotation-equity", "figure"),
    Output("rotation-summary", "data"),
    Output("rotation-attrib", "figure"),
    Output("rotation-blotter", "data"),
    Output("rotation-blotter-note", "children"),
    Input("rotation-date-range", "start_date"),
    Input("rotation-date-range", "end_date"),
    Input("rotation-smooth", "value"),
    Input("rotation-max-weight", "value"),
    Input("rotation-tc", "value"),
    Input("rotation-benchmark", "value"),
    Input("rotation-alloc", "value"),
    Input("rotation-options", "value"),
    Input("rotation-notional", "value"),
    Input("rotation-blotter-mode", "value"),
    Input("rotation-exec-date", "date"),
    Input("rotation-current-weights", "value"),
    Input("portfolio-selector", "value"),
)
def update_factor_rotation(
    start_date,
    end_date,
    smooth_lambda,
    max_weight,
    tc_bps,
    benchmark,
    alloc,
    options,
    notional,
    blotter_mode,
    exec_date,
    current_weights_mode,
    portfolio_id,
):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    options = options or []
    notional = float(notional or 100000)

    prices = load_ticker_prices(ETF_UNIVERSE, start=start, end=end)
    if prices.empty:
        empty = empty_figure("No data available for selected range.")
        return empty, empty, [], empty, [], None

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
        return empty, empty, [], empty, [], None

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

    core = load_portfolio_series(portfolio_id=portfolio_id or "schwab").loc[start:end]
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
    sleeve_notional = float(notional or 0.0) * float(alloc or 0.0)
    blotter_note = "Backtest uses full history; blotter shows the backtest rebalance timeline."

    if blotter_mode == "execution":
        blotter_note = "Backtest uses full history; blotter is filtered for execution starting at the selected date."
        exec_ts = pd.to_datetime(exec_date) if exec_date else EXECUTION_DEFAULT
        last_reb = weights.index[weights.index <= exec_ts].max() if not weights.empty else None
        if last_reb is None or pd.isna(last_reb):
            last_reb = weights.index.min()
        target = weights.loc[last_reb]
        if current_weights_mode == "last":
            prior_weights = target.copy()
        else:
            prior_weights = pd.Series(0.0, index=target.index)

        init_trade = target - prior_weights
        for asset, delta in init_trade.items():
            side = "Buy" if delta > 0 else "Sell" if delta < 0 else "Hold"
            blotter_rows.append({
                "date": exec_ts.date().isoformat(),
                "asset": asset,
                "prior_weight": f"{prior_weights[asset]:.2%}",
                "target_weight": f"{target[asset]:.2%}",
                "trade": f"{delta:.2%}",
                "side": side,
                "dollar": f"{delta * sleeve_notional:,.0f}",
            })

        future_dates = weights.index[weights.index > last_reb]
        if future_dates.empty:
            blotter_note += " No future rebalance dates available yet."
        else:
            prev = target
            for dt in future_dates:
                target_next = weights.loc[dt]
                delta = target_next - prev
                for asset, change in delta.items():
                    side = "Buy" if change > 0 else "Sell" if change < 0 else "Hold"
                    blotter_rows.append({
                        "date": dt.date().isoformat(),
                        "asset": asset,
                        "prior_weight": f"{prev[asset]:.2%}",
                        "target_weight": f"{target_next[asset]:.2%}",
                        "trade": f"{change:.2%}",
                        "side": side,
                        "dollar": f"{change * sleeve_notional:,.0f}",
                    })
                prev = target_next
    else:
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
                    "dollar": f"{delta * sleeve_notional:,.0f}",
                })

    return weights_fig, equity_fig, summary, contrib_fig, blotter_rows, blotter_note


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
