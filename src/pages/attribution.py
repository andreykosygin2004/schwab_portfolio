import dash
from dash import html, dcc, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px

from analytics.attribution import (
    build_pm_memo,
    factor_period_contributions,
    time_series_attribution,
    top_contributors,
)
from analytics.constants import ANALYSIS_END, DEFAULT_START_DATE_ANALYSIS
from analytics.factors import fit_ols
from analytics.regimes import returns_from_prices
from analytics_macro import load_ticker_prices
from analytics.portfolio import load_portfolio_series, risk_free_warning
from utils.transactions import (
    build_positions_from_transactions,
    build_starting_positions_from_mv,
    compute_trade_summary,
)
from viz.plots import empty_figure

dash.register_page(__name__, path="/attribution", name="Attribution")

PORTFOLIO_SERIES = load_portfolio_series()
DEFAULT_START = DEFAULT_START_DATE_ANALYSIS
DEFAULT_END = ANALYSIS_END

FACTOR_OPTIONS = ["SPY", "QQQ", "HYG", "TLT", "USO", "UUP", "GLD", "TIP"]


layout = html.Div([
    html.Br(),
    html.H2("Attribution"),
    html.Div(
        risk_free_warning(),
        style={"color": "#b45309", "marginBottom": "8px"},
    ) if risk_free_warning() else html.Div(),
    html.P("Explain performance using holdings and factor contributions."),
    html.Br(),
    html.Br(),

    html.Div([
        html.Div([
            html.Label("Date range"),
            dcc.DatePickerRange(
                id="attr-date-range",
                min_date_allowed=PORTFOLIO_SERIES.index.min(),
                max_date_allowed=PORTFOLIO_SERIES.index.max(),
                start_date=DEFAULT_START.date(),
                end_date=DEFAULT_END.date(),
            ),
        ], style={"maxWidth": "360px"}),
        html.Div([
            html.Label("Frequency"),
            dcc.RadioItems(
                id="attr-frequency",
                options=[{"label": f, "value": f} for f in ["Daily", "Weekly"]],
                value="Daily",
                inline=True,
            ),
        ], style={"maxWidth": "240px"}),
        html.Div([
            html.Label("Top N"),
            dcc.Input(id="attr-top-n", type="number", min=5, max=30, step=1, value=16),
        ], style={"maxWidth": "120px"}),
    ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap"}),

    html.Br(),
    html.Br(),
    html.H3("Holdings Attribution (Time-series)"),
    html.Br(),
    html.P("Method: time-series weights attribution using lagged MV weights × returns."),
    html.P("Total Return uses trade cashflows: (realized + unrealized - invested) / invested."),
    html.P("Contribution uses trade P&L (realized + unrealized - invested) from transactions."),
    html.Div(id="attr-holdings-warning", style={"color": "#b45309", "marginBottom": "6px"}),
    dcc.Loading(dcc.Graph(id="attr-holdings-bar")),
    dash_table.DataTable(
        id="attr-holdings-table",
        columns=[
            {"name": "Holding", "id": "holding"},
            {"name": "Avg Weight", "id": "avg_weight"},
            {"name": "Total Return (trade-based)", "id": "ret"},
            {"name": "Contribution (trade P&L)", "id": "contrib"},
            {"name": "% of total", "id": "pct_total"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px"},
    ),
    html.Br(),
    dbc.Button("Show transaction inference", id="attr-debug-toggle", size="sm", color="secondary"),
    dbc.Collapse(
        dash_table.DataTable(
            id="attr-debug-table",
            columns=[
                {"name": "Date", "id": "date"},
                {"name": "Symbol", "id": "symbol"},
                {"name": "Action", "id": "action"},
                {"name": "Amount", "id": "amount"},
                {"name": "Price Used", "id": "price_used"},
                {"name": "Implied Fill", "id": "implied_fill"},
                {"name": "Shares", "id": "inferred_shares"},
                {"name": "Confidence", "id": "confidence"},
                {"name": "Price Source", "id": "price_source"},
            ],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "6px"},
        ),
        id="attr-debug-collapse",
        is_open=False,
    ),

    html.Br(),
    html.Hr(),
    html.Br(),
    html.H3("Top Factor Drivers (Selected Window)"),
    html.Br(),
    html.P("Period-focused factors; monthly aggregation over the selected window."),
    dcc.Loading(dcc.Graph(id="attr-factor-bars")),
    dcc.Loading(dcc.Graph(id="attr-factor-cum")),

    html.Br(),
    html.Hr(),
    html.Br(),
    html.H3("PM Memo"),
    html.Br(),
    html.Ul(id="attr-memo"),
])


@callback(
    Output("attr-holdings-warning", "children"),
    Output("attr-holdings-bar", "figure"),
    Output("attr-holdings-table", "data"),
    Output("attr-debug-table", "data"),
    Output("attr-factor-bars", "figure"),
    Output("attr-factor-cum", "figure"),
    Output("attr-memo", "children"),
    Input("attr-date-range", "start_date"),
    Input("attr-date-range", "end_date"),
    Input("attr-frequency", "value"),
    Input("attr-top-n", "value"),
)
def update_attribution(start_date, end_date, freq, top_n):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    top_n = int(top_n or 10)

    holdings_ts = pd.read_csv("data/holdings_timeseries.csv", parse_dates=["Date"], index_col="Date").sort_index()
    price_hist = pd.read_csv("data/historical_prices.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    transactions = pd.read_csv("data/schwab_transactions.csv")
    splits = pd.read_csv("data/stock_splits.csv")

    df = holdings_ts.loc[start:end]
    if df.empty:
        empty = empty_figure("No data available.")
        return "No data available.", empty, [], [], empty, empty, []

    mv_cols = [c for c in df.columns if c.startswith("MV_")]
    warning = ""
    if not mv_cols:
        warning = "No historical market value columns available; falling back to snapshot weights."
    else:
        warning = "Using time-series weights (lagged MV weights × returns)."

    total_value = df["total_value_clean"].copy()
    if "total_value_clean_rf" in df.columns:
        total_value = df["total_value_clean_rf"].copy()

    tx_dates = pd.to_datetime(transactions["Date"], errors="coerce")
    tx_buy_sell = transactions[transactions["Action"].str.lower().isin(["buy", "sell"])]
    tx_start = pd.to_datetime(tx_buy_sell["Date"], errors="coerce").min() if not tx_buy_sell.empty else start
    start_full = min(start, tx_start) if pd.notna(tx_start) else start

    debug_rows = []
    mv_df = pd.DataFrame()
    price_slice = pd.DataFrame()
    if mv_cols:
        mv_df = df[mv_cols]
        tickers = [c.replace("MV_", "") for c in mv_cols]
        price_slice = price_hist.loc[start_full:end, price_hist.columns.intersection(tickers)]
        if price_slice.empty:
            empty = empty_figure("No price data.")
            return warning, empty, [], [], empty, empty, []
        if freq == "Weekly":
            mv_df = mv_df.resample("W-FRI").last()
            price_slice = price_slice.resample("W-FRI").last()
            total_value = total_value.resample("W-FRI").last()

    attr_df = time_series_attribution(mv_df, price_slice, total_value) if not mv_df.empty else pd.DataFrame()
    starting_positions = build_starting_positions_from_mv(mv_df, price_slice, start_full)
    positions, debug_df = build_positions_from_transactions(
        transactions,
        price_slice,
        start_full,
        end,
        starting_positions,
        splits=splits,
    )
    debug_rows = debug_df.to_dict("records") if not debug_df.empty else []
    if not debug_df.empty:
        low_conf = (debug_df["confidence"] == "low").mean()
        if low_conf > 0.3:
            warning = f"{warning} Low confidence on {low_conf:.0%} of inferred trades."
    if not positions.empty:
        prices_aligned = price_slice.reindex(positions.index).ffill()
        if freq == "Weekly":
            positions = positions.resample("W-FRI").last()
            prices_aligned = prices_aligned.resample("W-FRI").last()
        returns = prices_aligned.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        returns = returns.loc[start:end]
        positions = positions.loc[returns.index]
        prices_aligned = prices_aligned.loc[returns.index]
        mv = positions * prices_aligned
        total = mv.sum(axis=1).replace(0, np.nan)
        weights = mv.div(total, axis=0).fillna(0.0)
        weights_lag = weights.shift(1).fillna(0.0)
        trade_summary = compute_trade_summary(transactions, price_slice, start_full, end, splits=splits)
        total_contrib = trade_summary["pnl"].reindex(positions.columns).fillna(0.0) if not trade_summary.empty else 0.0
        total_pnl = total_contrib.sum()
        avg_weight = weights_lag.mean(axis=0)
        total_return = (1 + returns).prod() - 1
        attr_df = pd.DataFrame({
            "avg_weight": avg_weight,
            "total_return": total_return,
            "contribution": total_contrib,
        })
        attr_df["pct_total"] = attr_df["contribution"] / total_pnl if total_pnl != 0 else 0.0
        warning = "Using transaction-inferred positions (buy/sell cycles tracked; MoneyLink excluded)."

        if not trade_summary.empty:
            attr_df["total_return"] = attr_df.index.to_series().map(trade_summary["total_return"]).combine_first(attr_df["total_return"])
    if attr_df.empty:
        latest = df[mv_cols].iloc[-1].fillna(0.0) if mv_cols else pd.Series(dtype=float)
        total = latest.sum()
        weights = (latest / total).rename(lambda x: x.replace("MV_", "")) if total > 0 else latest * 0.0
        tickers = weights.index.tolist()
        price_slice = price_hist.loc[start:end, price_hist.columns.intersection(tickers)]
        if price_slice.empty:
            empty = empty_figure("No price data.")
            return warning, empty, [], debug_rows, empty, empty, []
        returns = returns_from_prices(price_slice, freq=freq)
        total_ret = (1 + returns).prod() - 1
        contrib = weights * total_ret
        attr_df = pd.DataFrame({
            "avg_weight": weights,
            "total_return": total_ret,
            "contribution": contrib,
        })
        attr_df["pct_total"] = attr_df["contribution"] / attr_df["contribution"].sum() if attr_df["contribution"].sum() != 0 else 0.0

    if attr_df.empty:
        empty = empty_figure("Attribution unavailable.")
        return warning or "Attribution unavailable.", empty, [], debug_rows, empty, empty, []

    contrib_top = top_contributors(attr_df["contribution"], top_n)
    attr_top = attr_df.loc[contrib_top.index]

    bar_fig = px.bar(
        attr_top["contribution"].sort_values(),
        orientation="h",
        title="Top Contributors / Detractors",
    )
    bar_fig.update_layout(height=420)
    bar_fig.update_xaxes(title_text="Contribution ($, trade P&L)")

    table_rows = []
    for holding, row in attr_top.iterrows():
        table_rows.append({
            "holding": holding,
            "avg_weight": f"{row['avg_weight']:.1%}",
            "ret": f"{row['total_return']:.1%}",
            "contrib": f"${row['contribution']:.2f}",
            "pct_total": f"{row['pct_total']:.1%}",
        })

    factor_prices = load_ticker_prices(FACTOR_OPTIONS, start=start, end=end)
    factor_bars = empty_figure("No factor data.")
    factor_cum = empty_figure("No factor data.")
    memo = []
    if not factor_prices.empty:
        port = load_portfolio_series().loc[start:end]
        port_ret = returns_from_prices(port, freq=freq)
        factor_ret = returns_from_prices(factor_prices, freq=freq)
        aligned_idx = port_ret.index.intersection(factor_ret.index)
        port_ret = port_ret.reindex(aligned_idx)
        factor_ret = factor_ret.reindex(aligned_idx)
        res = fit_ols(port_ret, factor_ret)
        betas = res["betas"]
        contrib_df = factor_period_contributions(betas, factor_ret)
        if not contrib_df.empty:
            contrib_m = contrib_df.resample("M").sum()
            factor_bars = px.bar(contrib_m, title="Top Factor Drivers (Monthly, Window)")
            factor_bars.update_layout(barmode="relative", height=420)
            factor_bars.update_yaxes(tickformat=".1%")
            explained = contrib_df.sum(axis=1)
            residual = port_ret.reindex(explained.index) - explained
            cum_total = (1 + port_ret).cumprod() - 1
            cum_explained = (1 + explained).cumprod() - 1
            cum_resid = (1 + residual).cumprod() - 1
            factor_cum = px.line(
                pd.DataFrame({"Total": cum_total, "Explained": cum_explained, "Residual": cum_resid}),
                title="Cumulative Explained vs Residual (Window)",
            )
            factor_cum.update_layout(height=420)
            factor_cum.update_yaxes(tickformat=".1%")

            top_factor = betas.abs().sort_values(ascending=False).index[0] if not betas.empty else None
            contrib_series = attr_df["contribution"]
            # Use signed max/min to preserve detractor sign and selection.
            top_contrib_name = contrib_series.idxmax() if not contrib_series.empty else None
            top_detractor_name = contrib_series.idxmin() if not contrib_series.empty else None
            memo = build_pm_memo({
                "top_contrib": f"{top_contrib_name} ({contrib_series.loc[top_contrib_name]:.1%})" if top_contrib_name else None,
                "top_detractor": f"{top_detractor_name} ({contrib_series.loc[top_detractor_name]:.1%})" if top_detractor_name else None,
                "top_factor": f"{top_factor} (beta {betas[top_factor]:.2f})" if top_factor else None,
                "residual": f"{cum_resid.iloc[-1]:.1%}" if not cum_resid.empty else None,
            })

    memo_list = [html.Li(m) for m in memo] if memo else [html.Li("No memo available for selected window.")]
    return warning, bar_fig, table_rows, debug_rows, factor_bars, factor_cum, memo_list


@callback(
    Output("attr-debug-collapse", "is_open"),
    Input("attr-debug-toggle", "n_clicks"),
    State("attr-debug-collapse", "is_open"),
)
def toggle_attr_debug(n_clicks, is_open):
    if not n_clicks:
        return is_open
    return not is_open
