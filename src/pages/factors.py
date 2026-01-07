import dash
from dash import html, dcc, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analytics.factors import align_returns, factor_contributions, fit_ols, rolling_multifactor
from analytics.risk import compute_returns
from analytics_macro import load_portfolio_series, load_ticker_prices
from analytics.portfolio import risk_free_warning
from viz.plots import empty_figure

dash.register_page(__name__, path="/factors", name="Factor Attribution")

PORTFOLIO_SERIES = load_portfolio_series()
MIN_DATE = PORTFOLIO_SERIES.index.min()
MAX_DATE = PORTFOLIO_SERIES.index.max()
DEFAULT_END = MAX_DATE
DEFAULT_START = max(MIN_DATE, DEFAULT_END - pd.DateOffset(years=3))

FACTOR_OPTIONS = ["SPY", "QQQ", "HYG", "LQD", "TLT", "TIP", "GLD", "USO", "UUP"]
DEFAULT_FACTORS = ["SPY", "QQQ", "HYG", "TLT", "USO", "UUP"]
FREQ_OPTIONS = ["Daily", "Weekly"]
INFO_STYLE = {"cursor": "pointer", "textDecoration": "underline"}


layout = html.Div([
    html.Br(),
    html.H2("Factor Attribution"),
    html.Div(
        risk_free_warning(),
        style={"color": "#b45309", "marginBottom": "8px"},
    ) if risk_free_warning() else html.Div(),
    html.P(
        "Estimate how macro and benchmark factors explain portfolio returns, and what residual return remains."
    ),
    html.Br(),

    html.Div([
        html.Div([
            html.Label("Date range"),
            dcc.DatePickerRange(
                id="factor-date-range",
                min_date_allowed=MIN_DATE,
                max_date_allowed=MAX_DATE,
                start_date=DEFAULT_START.date(),
                end_date=DEFAULT_END.date(),
            ),
        ], style={"maxWidth": "360px"}),
        html.Div([
            html.Label("Frequency"),
            dcc.RadioItems(
                id="factor-frequency",
                options=[{"label": f, "value": f} for f in FREQ_OPTIONS],
                value="Daily",
                inline=True,
            ),
        ], style={"maxWidth": "260px"}),
        html.Div([
            html.Label("Factors"),
            dcc.Dropdown(
                id="factor-select",
                options=[{"label": f, "value": f} for f in FACTOR_OPTIONS],
                value=DEFAULT_FACTORS,
                multi=True,
            ),
        ], style={"minWidth": "320px"}),
    ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap"}),

    html.Br(),
    html.Hr(),
    html.H3("Multi-factor Regression Summary"),
    html.Br(),
    dash_table.DataTable(
        id="factor-summary-table",
        columns=[
            {"name": "Factor", "id": "factor"},
            {"name": "Beta", "id": "beta"},
            {"name": "Std Err", "id": "stderr"},
            {"name": "R^2", "id": "r2"},
            {"name": "Obs", "id": "n_obs"},
            {"name": "Alpha (ann)", "id": "alpha_ann"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px"},
    ),

    html.Br(),
    html.Hr(),
    html.H3("Rolling Betas"),
    html.Br(),
    dcc.Loading(dcc.Graph(id="rolling-betas-graph")),

    html.Br(),
    html.Hr(),
    html.H3("Model-based Factor Contributions"),
    html.Br(),
    html.P("Explained return by factor (monthly aggregation) from the multi-factor model."),
    dcc.Loading(dcc.Graph(id="factor-contrib-graph")),
    dcc.Loading(dcc.Graph(id="residual-graph")),

    html.Br(),
    html.Hr(),
    html.H3("Stability Diagnostics"),
    html.Br(),
    dcc.Loading(dcc.Graph(id="stability-graph")),
])


@callback(
    Output("factor-summary-table", "data"),
    Output("rolling-betas-graph", "figure"),
    Output("factor-contrib-graph", "figure"),
    Output("residual-graph", "figure"),
    Output("stability-graph", "figure"),
    Input("factor-date-range", "start_date"),
    Input("factor-date-range", "end_date"),
    Input("factor-frequency", "value"),
    Input("factor-select", "value"),
)
def update_factor_page(start_date, end_date, freq, factors):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    factors = factors or []

    port_prices = PORTFOLIO_SERIES.loc[start:end].dropna()
    if port_prices.empty:
        empty = empty_figure("No data available for selected range.")
        return [], empty, empty, empty, empty

    factor_prices = load_ticker_prices(factors, start=start, end=end)
    if factor_prices.empty:
        empty = empty_figure("No factor data available.")
        return [], empty, empty, empty, empty

    port_ret = compute_returns(port_prices, freq)
    factor_ret = compute_returns(factor_prices, freq)
    port_ret, factor_ret = align_returns(port_ret, factor_ret)
    if port_ret.empty or factor_ret.empty:
        empty = empty_figure("Not enough aligned data.")
        return [], empty, empty, empty, empty

    fit = fit_ols(port_ret, factor_ret)
    betas = fit["betas"]
    r2 = fit["r2"]
    stderr = fit["stderr"]
    n_obs = len(port_ret)
    periods = 252 if freq == "Daily" else 52
    alpha_ann = fit["alpha"] * periods if fit["alpha"] == fit["alpha"] else np.nan

    summary_rows = []
    for i, factor in enumerate(betas.index):
        se = stderr[i + 1] if stderr is not None and len(stderr) > i + 1 else np.nan
        summary_rows.append({
            "factor": factor,
            "beta": f"{betas[factor]:.2f}",
            "stderr": f"{se:.3f}" if pd.notna(se) else "n/a",
            "r2": f"{r2:.2f}",
            "n_obs": n_obs,
            "alpha_ann": f"{alpha_ann:.2%}" if pd.notna(alpha_ann) else "n/a",
        })

    window = 63 if freq == "Daily" else 26
    rolling_betas, rolling_r2 = rolling_multifactor(port_ret, factor_ret, window=window)
    rolling_fig = empty_figure("Not enough data for rolling betas.")
    if not rolling_betas.empty:
        rolling_fig = px.line(rolling_betas, title=f"Rolling Betas (window={window})")
        rolling_fig.update_layout(height=450, legend_title_text="")

    contrib = factor_contributions(rolling_betas, factor_ret)
    contrib_fig = empty_figure("Not enough data for contributions.")
    residual_fig = empty_figure("Not enough data for residuals.")
    if not contrib.empty:
        contrib_resampled = contrib.resample("M").sum()
        contrib_fig = px.bar(
            contrib_resampled,
            title="Model-based Factor Contribution (Monthly, Explained Return)",
            labels={"value": "Contribution", "index": "Date"},
        )
        contrib_fig.update_layout(barmode="relative", height=450)
        contrib_fig.update_yaxes(tickformat=".1%")

        explained = contrib.sum(axis=1)
        residual = port_ret.reindex(explained.index) - explained
        cumulative_residual = (1 + residual).cumprod() - 1
        residual_fig = go.Figure()
        residual_fig.add_trace(go.Scatter(x=residual.index, y=residual.values, name="Residual"))
        residual_fig.add_trace(go.Scatter(x=cumulative_residual.index, y=cumulative_residual.values, name="Cumulative Residual"))
        residual_fig.update_layout(title="Model Residual (Period) and Cumulative Residual", height=450, legend_title_text="")
        residual_fig.update_yaxes(tickformat=".1%")

    stability_fig = empty_figure("Not enough data for stability diagnostics.")
    if not rolling_betas.empty:
        beta_dispersion = rolling_betas.std(axis=1)
        stability_fig = go.Figure()
        stability_fig.add_trace(go.Scatter(x=rolling_r2.index, y=rolling_r2.values, name="Rolling R^2"))
        stability_fig.add_trace(go.Scatter(x=beta_dispersion.index, y=beta_dispersion.values, name="Beta Dispersion", yaxis="y2"))
        stability_fig.update_layout(
            title="Rolling R^2 and Beta Dispersion",
            height=450,
            legend_title_text="",
            yaxis=dict(title="R^2"),
            yaxis2=dict(title="Beta Dispersion", overlaying="y", side="right"),
        )

    return summary_rows, rolling_fig, contrib_fig, residual_fig, stability_fig
