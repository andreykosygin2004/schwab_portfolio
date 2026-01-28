import dash
from dash import html, dcc, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analytics.constants import DEFAULT_START_DATE_ANALYSIS
from analytics.common import annualize_return_cagr, annualize_vol
from analytics.portfolio import get_portfolio_date_bounds, load_portfolio_series
from analytics.regimes import compute_regime_features, label_regimes
from analytics_macro import load_ticker_prices
from viz.plots import empty_figure
from strategies.factor_rotation import (
    RotationParams,
    backtest_rotation,
    compute_target_weights,
    ETF_UNIVERSE,
    monthly_returns as rotation_monthly_returns,
)
from utils.sleeve_correlation_aware import (
    backtest_with_weights,
    compute_correlation_aware_weights,
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
            dbc.Row([
                dbc.Col([
                    html.Label("Sleeve Mode"),
                    dcc.RadioItems(
                        id="rotation-mode",
                        options=[
                            {"label": "Standalone (current)", "value": "standalone"},
                            {"label": "Correlation-aware (new)", "value": "corr"},
                        ],
                        value="standalone",
                        inline=True,
                    ),
                ], xs=12, md=5),
                dbc.Col([
                    html.Label("Core portfolio for correlation"),
                    dcc.Dropdown(
                        id="rotation-core-portfolio",
                        options=[
                            {"label": "My Portfolio", "value": "schwab"},
                            {"label": "Algory", "value": "algory"},
                        ],
                        value="schwab",
                        clearable=False,
                    ),
                ], xs=12, md=3),
                dbc.Col([
                    html.Label("Correlation penalty (gamma)"),
                    dcc.Slider(
                        id="rotation-corr-gamma",
                        min=0.0,
                        max=5.0,
                        step=0.25,
                        value=2.0,
                        marks={0: "0", 2: "2", 5: "5"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], xs=12, md=4),
            ], className="g-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Correlation lookback (months)"),
                    dcc.Slider(
                        id="rotation-corr-lookback",
                        min=3,
                        max=18,
                        step=1,
                        value=6,
                        marks={3: "3", 6: "6", 12: "12", 18: "18"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], xs=12, md=6),
                dbc.Col([
                    html.Label("Use absolute correlation"),
                    dcc.Checklist(
                        id="rotation-corr-abs",
                        options=[{"label": "Abs corr", "value": "abs"}],
                        value=["abs"],
                        inline=True,
                    ),
                ], xs=12, md=6),
            ], className="g-3"),
        ]),
        style={"borderRadius": "10px", "border": "1px solid #e6e9ee"},
    ),

    html.Br(),
    dcc.Loading(dcc.Graph(id="rotation-weights")),
    html.Br(),
    html.H5("Correlation Diagnostics (Correlation-aware mode)"),
    dash_table.DataTable(
        id="rotation-corr-table",
        columns=[
            {"name": "Asset", "id": "asset"},
            {"name": "Corr", "id": "corr"},
            {"name": "Penalty", "id": "penalty"},
        ],
        style_table={"overflowX": "auto", "maxWidth": "520px"},
        style_cell={"textAlign": "left", "padding": "6px"},
    ),
    dcc.Loading(dcc.Graph(id="rotation-corr-rolling")),
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
    Output("rotation-date-range", "min_date_allowed"),
    Output("rotation-date-range", "max_date_allowed"),
    Output("rotation-date-range", "start_date"),
    Output("rotation-date-range", "end_date"),
    Output("rotation-exec-date", "min_date_allowed"),
    Output("rotation-exec-date", "max_date_allowed"),
    Output("rotation-exec-date", "date"),
    Input("portfolio-selector", "value"),
)
def update_rotation_date_range(portfolio_id):
    port_min, port_max = get_portfolio_date_bounds(portfolio_id or "schwab")
    if port_min is None or port_max is None:
        return MIN_DATE, MAX_DATE, DEFAULT_START.date(), DEFAULT_END.date(), MIN_DATE, MAX_DATE, EXECUTION_DEFAULT.date()
    if portfolio_id == "algory":
        return port_min, port_max, port_min.date(), port_max.date(), port_min, port_max, port_max.date()
    return port_min, port_max, DEFAULT_START.date(), port_max.date(), port_min, port_max, EXECUTION_DEFAULT.date()


@callback(
    Output("rotation-weights", "figure"),
    Output("rotation-equity", "figure"),
    Output("rotation-summary", "data"),
    Output("rotation-attrib", "figure"),
    Output("rotation-blotter", "data"),
    Output("rotation-blotter-note", "children"),
    Output("rotation-corr-table", "data"),
    Output("rotation-corr-rolling", "figure"),
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
    Input("rotation-mode", "value"),
    Input("rotation-core-portfolio", "value"),
    Input("rotation-corr-gamma", "value"),
    Input("rotation-corr-lookback", "value"),
    Input("rotation-corr-abs", "value"),
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
    sleeve_mode,
    core_portfolio,
    corr_gamma,
    corr_lookback,
    corr_abs,
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
        return empty, empty, [], empty, [], None, [], empty

    weights = results["weights"]
    returns = results["returns"]
    equity = results["equity"]
    turnover = results["turnover"]
    monthly_rets = results["monthly_returns"]

    corr_table = []
    corr_fig = empty_figure("Enable correlation-aware mode to view diagnostics.")
    corr_weights = pd.DataFrame()
    corr_returns = pd.Series(dtype=float)
    corr_equity = pd.Series(dtype=float)
    if sleeve_mode == "corr":
        base_targets = compute_target_weights(prices, params, regime_labels)
        asset_rets = rotation_monthly_returns(prices).dropna(how="all")
        core_series = load_portfolio_series(portfolio_id=core_portfolio or "schwab")
        core_m = core_series.resample("M").last().pct_change().replace([np.inf, -np.inf], np.nan)
        core_m = core_m.reindex(asset_rets.index).fillna(0.0)
        corr_weights, diag = compute_correlation_aware_weights(
            base_targets,
            asset_rets,
            core_m,
            int(corr_lookback or 6),
            float(corr_gamma or 0.0),
            "abs" in (corr_abs or []),
            float(max_weight or 0.45),
            float(smooth_lambda or 0.35),
        )
        if not corr_weights.empty:
            corr_bt = backtest_with_weights(corr_weights, asset_rets, float(tc_bps or 0.0))
            corr_returns = corr_bt.get("returns", pd.Series(dtype=float))
            corr_equity = corr_bt.get("equity", pd.Series(dtype=float))
            if not diag.empty:
                last_date = diag["date"].max()
                view = diag[diag["date"] == last_date].copy()
                view = view.sort_values("corr", ascending=False)
                corr_table = [
                    {
                        "asset": row["asset"],
                        "corr": f"{row['corr']:.2f}",
                        "penalty": f"{row['penalty']:.2f}",
                    }
                    for _, row in view.iterrows()
                ]
            if not corr_returns.empty and not core_m.empty:
                aligned = pd.concat([corr_returns, core_m], axis=1).dropna()
                if not aligned.empty:
                    rolling = aligned.iloc[:, 0].rolling(int(corr_lookback or 6)).corr(aligned.iloc[:, 1])
                    corr_fig = go.Figure()
                    corr_fig.add_trace(go.Scatter(x=rolling.index, y=rolling.values, name="Sleeve vs Core"))
                    corr_fig.update_layout(
                        title="Effective Sleeve Correlation vs Core (Rolling)",
                        height=320,
                        legend_title_text="",
                    )

    weights_fig = empty_figure("No weights to display.")
    if not weights.empty:
        if sleeve_mode == "corr" and not corr_weights.empty:
            weights_fig = px.area(corr_weights, title="Target Weights Over Time (Correlation-aware)")
        else:
            weights_fig = px.area(weights, title="Target Weights Over Time (Standalone)")
        weights_fig.update_layout(height=420, legend_title_text="")
        weights_fig.update_yaxes(tickformat=".0%")

    bench_prices = load_ticker_prices([benchmark], start=weights.index.min(), end=weights.index.max())
    bench_ret = rotation_monthly_returns(bench_prices[[benchmark]]) if not bench_prices.empty else pd.DataFrame()
    bench_ret = bench_ret[benchmark].reindex(returns.index).fillna(0.0) if not bench_ret.empty else pd.Series(dtype=float)

    core = load_portfolio_series(portfolio_id=portfolio_id or "schwab").loc[start:end]
    core_m = core.resample("M").last().pct_change().replace([np.inf, -np.inf], np.nan)
    core_m = core_m.reindex(returns.index).fillna(0.0)
    if sleeve_mode == "corr" and not corr_returns.empty:
        blended = (1 - float(alloc or 0.0)) * core_m + float(alloc or 0.0) * corr_returns.reindex(core_m.index).fillna(0.0)
    else:
        blended = (1 - float(alloc or 0.0)) * core_m + float(alloc or 0.0) * returns

    equity_fig = go.Figure()
    equity_fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Standalone Sleeve"))
    if sleeve_mode == "corr" and not corr_equity.empty:
        equity_fig.add_trace(go.Scatter(x=corr_equity.index, y=corr_equity.values, name="Correlation-aware Sleeve"))
    if not bench_ret.empty:
        equity_fig.add_trace(go.Scatter(x=bench_ret.index, y=(1 + bench_ret).cumprod(), name=benchmark))
    equity_fig.add_trace(go.Scatter(x=blended.index, y=(1 + blended).cumprod(), name="Blended Core+Sleeve (selected)"))
    equity_fig.update_layout(title="Equity Curve (Monthly)", height=420, legend_title_text="")

    use_returns = corr_returns if sleeve_mode == "corr" and not corr_returns.empty else returns
    use_equity = corr_equity if sleeve_mode == "corr" and not corr_equity.empty else equity
    use_turnover = turnover
    if sleeve_mode == "corr" and not corr_returns.empty:
        use_turnover = backtest_with_weights(corr_weights, monthly_rets, float(tc_bps or 0.0)).get("turnover", turnover)

    ann_return = annualize_return_cagr(use_returns, 12)
    ann_vol = annualize_vol(use_returns, 12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    dd = (use_equity / use_equity.cummax() - 1).min()
    calmar = ann_return / abs(dd) if dd < 0 else np.nan
    te = (use_returns - bench_ret).std() * np.sqrt(12) if not bench_ret.empty else np.nan
    ir = (use_returns - bench_ret).mean() / (use_returns - bench_ret).std() * np.sqrt(12) if not bench_ret.empty else np.nan

    summary = [
        {"metric": "Mode", "value": "Correlation-aware" if sleeve_mode == "corr" else "Standalone"},
        {"metric": "CAGR", "value": f"{ann_return:.2%}"},
        {"metric": "Ann Vol", "value": f"{ann_vol:.2%}"},
        {"metric": "Sharpe (Rf=0)", "value": f"{sharpe:.2f}" if pd.notna(sharpe) else "n/a"},
        {"metric": "Max DD", "value": f"{dd:.2%}"},
        {"metric": "Calmar", "value": f"{calmar:.2f}" if pd.notna(calmar) else "n/a"},
        {"metric": "Avg Monthly Turnover", "value": f"{use_turnover.mean():.2%}"},
        {"metric": f"Tracking Error vs {benchmark}", "value": f"{te:.2%}" if pd.notna(te) else "n/a"},
        {"metric": f"Info Ratio vs {benchmark}", "value": f"{ir:.2f}" if pd.notna(ir) else "n/a"},
    ]

    contrib_weights = results["weights_lag"]
    if sleeve_mode == "corr" and not corr_weights.empty:
        contrib_weights = corr_weights.shift(1).reindex(monthly_rets.index).fillna(0.0)
    contrib = (contrib_weights * monthly_rets).reindex(returns.index)
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
    blotter_weights = corr_weights if sleeve_mode == "corr" and not corr_weights.empty else weights

    if blotter_mode == "execution":
        blotter_note = "Backtest uses full history; blotter is filtered for execution starting at the selected date."
        if sleeve_mode == "corr":
            blotter_note += " (Correlation-aware)"
        exec_ts = pd.to_datetime(exec_date) if exec_date else EXECUTION_DEFAULT
        last_reb = blotter_weights.index[blotter_weights.index <= exec_ts].max() if not blotter_weights.empty else None
        if last_reb is None or pd.isna(last_reb):
            last_reb = blotter_weights.index.min()
        target = blotter_weights.loc[last_reb]
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

        future_dates = blotter_weights.index[blotter_weights.index > last_reb]
        if future_dates.empty:
            blotter_note += " No future rebalance dates available yet."
        else:
            prev = target
            for dt in future_dates:
                target_next = blotter_weights.loc[dt]
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
        weights_diff = blotter_weights.diff().fillna(blotter_weights.iloc[0])
        for dt, row in weights_diff.iterrows():
            for asset, delta in row.items():
                prior = blotter_weights.loc[dt, asset] - delta
                side = "Buy" if delta > 0 else "Sell" if delta < 0 else "Hold"
                blotter_rows.append({
                    "date": dt.date().isoformat(),
                    "asset": asset,
                    "prior_weight": f"{prior:.2%}",
                    "target_weight": f"{blotter_weights.loc[dt, asset]:.2%}",
                    "trade": f"{delta:.2%}",
                    "side": side,
                    "dollar": f"{delta * sleeve_notional:,.0f}",
                })

    return weights_fig, equity_fig, summary, contrib_fig, blotter_rows, blotter_note, corr_table, corr_fig


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
