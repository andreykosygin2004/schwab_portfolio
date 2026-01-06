import dash
from dash import html, dcc, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px

from analytics.attribution import build_pm_memo, compute_contributions, factor_period_contributions, top_contributors
from analytics.constants import ANALYSIS_END, ANALYSIS_START
from analytics.factors import fit_ols
from analytics.regimes import returns_from_prices
from analytics_macro import load_ticker_prices
from analytics.portfolio import load_portfolio_series, risk_free_warning
from viz.plots import empty_figure

dash.register_page(__name__, path="/attribution", name="Attribution")

PORTFOLIO_SERIES = load_portfolio_series()
DEFAULT_START = ANALYSIS_START
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
            dcc.Input(id="attr-top-n", type="number", min=5, max=30, step=1, value=10),
        ], style={"maxWidth": "120px"}),
    ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap"}),

    html.Br(),
    html.H3("Holdings Attribution"),
    html.Div(id="attr-holdings-warning", style={"color": "#b45309", "marginBottom": "6px"}),
    dcc.Loading(dcc.Graph(id="attr-holdings-bar")),
    dash_table.DataTable(
        id="attr-holdings-table",
        columns=[
            {"name": "Holding", "id": "holding"},
            {"name": "Weight", "id": "weight"},
            {"name": "Return", "id": "ret"},
            {"name": "Contribution", "id": "contrib"},
            {"name": "% of total", "id": "pct_total"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px"},
    ),

    html.Br(),
    html.H3("Factor Attribution"),
    dcc.Loading(dcc.Graph(id="attr-factor-bars")),
    dcc.Loading(dcc.Graph(id="attr-factor-cum")),

    html.Br(),
    html.H3("PM Memo"),
    html.Ul(id="attr-memo"),
])


@callback(
    Output("attr-holdings-warning", "children"),
    Output("attr-holdings-bar", "figure"),
    Output("attr-holdings-table", "data"),
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

    df = holdings_ts.loc[start:end]
    if df.empty:
        empty = empty_figure("No data available.")
        return "No data available.", empty, [], empty, empty, []

    mv_cols = [c for c in df.columns if c.startswith("MV_")]
    warning = ""
    if not mv_cols:
        warning = "No historical weights available; using current holdings weights."
    weights = None
    if mv_cols:
        latest = df[mv_cols].iloc[-1].fillna(0.0)
        total = latest.sum()
        weights = (latest / total).rename(lambda x: x.replace("MV_", "")) if total > 0 else latest * 0.0
    else:
        weights = pd.Series(dtype=float)

    tickers = weights.index.tolist()
    price_slice = price_hist.loc[start:end, price_hist.columns.intersection(tickers)]
    if price_slice.empty:
        empty = empty_figure("No price data.")
        return warning, empty, [], empty, empty, []

    returns = returns_from_prices(price_slice, freq=freq)
    total_ret = (1 + returns).prod() - 1
    contrib = compute_contributions(weights, total_ret)
    contrib_top = top_contributors(contrib, top_n)
    contrib_pct = contrib_top / contrib.sum() if contrib.sum() != 0 else contrib_top * 0.0

    bar_fig = px.bar(
        contrib_top.sort_values(),
        orientation="h",
        title="Top Contributors / Detractors",
    )
    bar_fig.update_layout(height=420)
    bar_fig.update_xaxes(title_text="Contribution", tickformat=".1%")

    table_rows = []
    for holding, value in contrib_top.items():
        table_rows.append({
            "holding": holding,
            "weight": f"{weights.get(holding, 0.0):.1%}",
            "ret": f"{total_ret.get(holding, 0.0):.1%}",
            "contrib": f"{value:.1%}",
            "pct_total": f"{contrib_pct.get(holding, 0.0):.1%}",
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
            factor_bars = px.bar(contrib_m, title="Factor Contributions (Monthly)")
            factor_bars.update_layout(barmode="relative", height=420)
            factor_bars.update_yaxes(tickformat=".1%")
            explained = contrib_df.sum(axis=1)
            residual = port_ret.reindex(explained.index) - explained
            cum_total = (1 + port_ret).cumprod() - 1
            cum_explained = (1 + explained).cumprod() - 1
            cum_resid = (1 + residual).cumprod() - 1
            factor_cum = px.line(
                pd.DataFrame({"Total": cum_total, "Explained": cum_explained, "Residual": cum_resid}),
                title="Cumulative Explained vs Residual",
            )
            factor_cum.update_layout(height=420)
            factor_cum.update_yaxes(tickformat=".1%")

            top_factor = betas.abs().sort_values(ascending=False).index[0] if not betas.empty else None
            memo = build_pm_memo({
                "top_contrib": f"{contrib_top.index[0]} ({contrib_top.iloc[0]:.1%})" if not contrib_top.empty else None,
                "top_detractor": f"{contrib_top.index[-1]} ({contrib_top.iloc[-1]:.1%})" if not contrib_top.empty else None,
                "top_factor": f"{top_factor} (beta {betas[top_factor]:.2f})" if top_factor else None,
                "residual": f"{cum_resid.iloc[-1]:.1%}" if not cum_resid.empty else None,
            })

    memo_list = [html.Li(m) for m in memo] if memo else [html.Li("No memo available for selected window.")]
    return warning, bar_fig, table_rows, factor_bars, factor_cum, memo_list
