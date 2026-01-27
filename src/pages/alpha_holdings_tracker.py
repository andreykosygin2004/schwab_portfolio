import dash
from dash import html, dcc, Input, Output, State, callback, dash_table
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analytics.constants import DEFAULT_START_DATE_ANALYSIS
from analytics.portfolio import (
    build_portfolio_timeseries,
    get_portfolio_date_bounds,
    load_holdings_timeseries,
    load_portfolio_series,
)
from analytics_macro import compute_returns, load_ticker_prices
from utils.alpha_decompose import (
    compute_holding_alpha_contrib,
    compute_portfolio_explained_residual,
    fit_factor_model,
    fit_factor_model_rolling,
)
from viz.plots import empty_figure


dash.register_page(__name__, path="/alpha-holdings", name="Alpha Holdings Tracker")

DEFAULT_FACTORS = ["QQQ", "HYG", "TLT", "GLD"]

layout = html.Div([
    html.Br(),
    html.H2("Alpha Holdings Tracker"),
    html.P("Decompose each holding’s return into explained vs residual (alpha proxy) relative to selected factors."),
    html.Br(),
    html.Br(),

    html.Div([
        html.Label("Portfolio"),
        dcc.Dropdown(
            id="alpha-portfolio",
            options=[
                {"label": "My Portfolio", "value": "schwab"},
                {"label": "Algory", "value": "algory"},
            ],
            value="schwab",
            clearable=False,
        ),
    ], style={"maxWidth": "260px"}),
    html.Br(),

    html.Div([
        html.Div([
            html.Label("Date range"),
            dcc.DatePickerRange(
                id="alpha-date-range",
                start_date=DEFAULT_START_DATE_ANALYSIS,
            ),
        ], style={"display": "inline-block", "marginRight": "18px"}),
        html.Div([
            html.Label("Frequency"),
            dcc.RadioItems(
                id="alpha-freq",
                options=[{"label": "Daily", "value": "Daily"}, {"label": "Weekly", "value": "Weekly"}],
                value="Daily",
                inline=True,
            ),
        ], style={"display": "inline-block", "marginRight": "18px"}),
        html.Div([
            html.Label("Factors"),
            dcc.Dropdown(
                id="alpha-factors",
                options=[{"label": f, "value": f} for f in ["SPY", "QQQ", "HYG", "TLT", "GLD", "USO", "TIP", "UUP"]],
                value=DEFAULT_FACTORS,
                multi=True,
            ),
        ], style={"display": "inline-block", "minWidth": "320px"}),
    ]),
    html.Br(),
    html.Div([
        html.Div([
            dcc.Checklist(
                id="alpha-rolling",
                options=[{"label": "Rolling regression", "value": "rolling"}],
                value=[],
                inline=True,
            ),
        ], style={"display": "inline-block", "marginRight": "18px"}),
        html.Div([
            html.Label("Rolling window (periods)"),
            dcc.Input(id="alpha-rolling-window", type="number", value=52, min=12, step=1),
        ], style={"display": "inline-block"}),
    ]),
    html.Br(),
    html.Button("Recompute", id="alpha-recompute"),
    html.Br(),
    html.Br(),

    dcc.Loading(dcc.Graph(id="alpha-explained")),
    dcc.Loading(dcc.Graph(id="alpha-top-holdings")),
    html.P("Table values are additive (sum of period contributions).", style={"color": "#5b6675"}),
    dash_table.DataTable(
        id="alpha-table",
        columns=[
            {"name": "Holding", "id": "holding"},
            {"name": "Avg Weight", "id": "avg_weight"},
            {"name": "Cum Alpha Contrib", "id": "alpha_contrib"},
            {"name": "% of Total Alpha", "id": "pct_total"},
            {"name": "Residual Vol", "id": "resid_vol"},
            {"name": "Residual IR", "id": "resid_ir"},
            {"name": "Betas", "id": "betas"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px"},
    ),
    html.Br(),
])


@callback(
    Output("alpha-date-range", "min_date_allowed"),
    Output("alpha-date-range", "max_date_allowed"),
    Output("alpha-date-range", "start_date"),
    Output("alpha-date-range", "end_date"),
    Input("alpha-portfolio", "value"),
)
def update_alpha_date_range(portfolio_id):
    min_dt, max_dt = get_portfolio_date_bounds(portfolio_id or "schwab")
    if min_dt is None or max_dt is None:
        return None, None, DEFAULT_START_DATE_ANALYSIS, None
    start = max(min_dt, pd.to_datetime(DEFAULT_START_DATE_ANALYSIS))
    return min_dt, max_dt, start.date(), max_dt.date()


@callback(
    Output("alpha-explained", "figure"),
    Output("alpha-top-holdings", "figure"),
    Output("alpha-table", "data"),
    Input("alpha-recompute", "n_clicks"),
    State("alpha-date-range", "start_date"),
    State("alpha-date-range", "end_date"),
    State("alpha-freq", "value"),
    State("alpha-factors", "value"),
    State("alpha-rolling", "value"),
    State("alpha-rolling-window", "value"),
    State("alpha-portfolio", "value"),
)
def update_alpha_holdings(
    n_clicks,
    start_date,
    end_date,
    freq,
    factors,
    rolling_mode,
    rolling_window,
    portfolio_id,
):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    factors = [f for f in (factors or []) if f]
    if not factors:
        empty = empty_figure("Select at least one factor.")
        return empty, empty, []

    holdings_ts = load_holdings_timeseries(portfolio_id=portfolio_id or "schwab")
    if holdings_ts.empty:
        empty = empty_figure("No holdings time series available.")
        return empty, empty, []
    mv_cols = [c for c in holdings_ts.columns if c.startswith("MV_")]
    if not mv_cols:
        empty = empty_figure("No market value columns available.")
        return empty, empty, []
    mv = holdings_ts[mv_cols].loc[start:end].copy()
    mv.columns = [c.replace("MV_", "") for c in mv.columns]
    portfolio_ts = build_portfolio_timeseries(portfolio_id=portfolio_id or "schwab")
    cash_mv = pd.Series(dtype=float)
    if not portfolio_ts.empty:
        cash_mv = portfolio_ts.get("cash_value_clean_rf")
        if cash_mv is None:
            cash_mv = portfolio_ts.get("cash_value_rf")
        if cash_mv is None:
            cash_mv = portfolio_ts.get("cash_balance_clean", pd.Series(dtype=float))
        cash_mv = cash_mv.loc[start:end].reindex(mv.index).fillna(0.0)
        mv["CASH"] = cash_mv
    mv = mv.loc[:, (mv.sum(axis=0) > 0)]
    if mv.empty:
        empty = empty_figure("No holdings market values in selected window.")
        return empty, empty, []

    tickers = [t for t in mv.columns if t != "CASH"]
    prices = load_ticker_prices(tickers, start=start, end=end) if tickers else pd.DataFrame()
    if tickers and prices.empty:
        empty = empty_figure("Missing holdings price history.")
        return empty, empty, []

    prices = prices.reindex(mv.index).ffill() if not prices.empty else prices
    hold_ret = compute_returns(prices, freq) if not prices.empty else pd.DataFrame()
    if hold_ret.empty:
        empty = empty_figure("Not enough holding returns.")
        return empty, empty, []

    if freq == "Weekly":
        mv = mv.resample("W-FRI").last()
    total_mv = mv.sum(axis=1).replace(0, np.nan)
    weights = mv.div(total_mv, axis=0).fillna(0.0)
    weights = weights.reindex(hold_ret.index).fillna(0.0)

    factor_prices = load_ticker_prices(factors, start=start, end=end)
    factor_ret = compute_returns(factor_prices, freq)
    factor_ret = factor_ret.reindex(hold_ret.index).dropna(how="all")
    hold_ret = hold_ret.reindex(factor_ret.index).dropna(how="all")
    weights = weights.reindex(hold_ret.index).fillna(0.0)
    if hold_ret.empty or factor_ret.empty:
        empty = empty_figure("Not enough aligned data for factor decomposition.")
        return empty, empty, []

    rolling = "rolling" in (rolling_mode or [])
    window = int(rolling_window or 52)

    residuals_df = pd.DataFrame(index=hold_ret.index, columns=hold_ret.columns, dtype=float)
    betas = {}
    for col in hold_ret.columns:
        series = hold_ret[col].dropna()
        factors_aligned = factor_ret.reindex(series.index).dropna(how="any")
        series = series.reindex(factors_aligned.index)
        if series.empty or factors_aligned.empty:
            continue
        if rolling and len(series) >= window:
            resid, beta_df = fit_factor_model_rolling(series, factors_aligned, window)
            residuals_df.loc[resid.index, col] = resid
            if not beta_df.empty:
                betas[col] = beta_df.iloc[-1]
        else:
            fit = fit_factor_model(series, factors_aligned)
            residuals_df.loc[fit["residuals"].index, col] = fit["residuals"]
            betas[col] = fit["betas"]

    residuals_df = residuals_df.dropna(how="all")
    if residuals_df.empty:
        empty = empty_figure("No residuals computed (check factor selection).")
        return empty, empty, []

    contrib_df = compute_holding_alpha_contrib(weights, residuals_df)
    cum_alpha = contrib_df.sum(axis=0)
    cum_alpha = cum_alpha.reindex(residuals_df.columns).dropna()
    cum_alpha = cum_alpha.sort_values(ascending=False)
    total_alpha = cum_alpha.sum()
    avg_weight = weights.shift(1).mean(axis=0)

    periods = 252 if freq == "Daily" else 52
    table_rows = []
    for holding, value in cum_alpha.items():
        resid_series = residuals_df[holding].dropna()
        resid_vol = resid_series.std() * np.sqrt(periods) if not resid_series.empty else np.nan
        resid_ir = (resid_series.mean() / resid_series.std()) * np.sqrt(periods) if resid_series.std() > 0 else np.nan
        beta_str = ""
        if holding in betas and isinstance(betas[holding], pd.Series):
            beta_str = ", ".join([f"{k}:{v:.2f}" for k, v in betas[holding].items()])
        table_rows.append({
            "holding": holding,
            "avg_weight": f"{avg_weight.get(holding, 0.0):.1%}",
            "alpha_contrib": f"{value:.2%}",
            "pct_total": f"{(value / total_alpha):.1%}" if total_alpha != 0 else "0.0%",
            "resid_vol": f"{resid_vol:.2%}" if pd.notna(resid_vol) else "n/a",
            "resid_ir": f"{resid_ir:.2f}" if pd.notna(resid_ir) else "n/a",
            "betas": beta_str,
        })

    top_holdings = cum_alpha.head(10)
    top_fig = px.bar(
        top_holdings[::-1],
        orientation="h",
        title="Top Alpha Contributors (Holdings)",
    )
    top_fig.update_layout(height=420, legend_title_text="")
    top_fig.update_xaxes(tickformat=".1%")

    port_series = load_portfolio_series(portfolio_id=portfolio_id or "schwab").loc[start:end]
    port_returns = compute_returns(port_series.to_frame("portfolio"), freq)["portfolio"]
    port_returns = port_returns.reindex(factor_ret.index).dropna()
    explained, residual = compute_portfolio_explained_residual(port_returns, factor_ret)
    explained = explained.reindex(port_returns.index)
    residual = residual.reindex(port_returns.index)
    explained_cum = (1 + explained).cumprod() - 1 if not explained.empty else pd.Series(dtype=float)
    residual_cum = (1 + residual).cumprod() - 1 if not residual.empty else pd.Series(dtype=float)
    total_cum = (1 + port_returns).cumprod() - 1 if not port_returns.empty else pd.Series(dtype=float)

    explained_fig = empty_figure("Not enough data for portfolio decomposition.")
    if not explained_cum.empty:
        explained_fig = go.Figure()
        explained_fig.add_trace(go.Scatter(x=total_cum.index, y=total_cum.values, name="Total"))
        explained_fig.add_trace(go.Scatter(x=explained_cum.index, y=explained_cum.values, name="Explained"))
        explained_fig.add_trace(go.Scatter(x=residual_cum.index, y=residual_cum.values, name="Residual (Alpha proxy)"))
        explained_fig.update_layout(
            title="Cumulative Total vs Explained vs Residual (Portfolio)",
            height=420,
            legend_title_text="",
        )
        explained_fig.update_yaxes(tickformat=".1%")

    return explained_fig, top_fig, table_rows
