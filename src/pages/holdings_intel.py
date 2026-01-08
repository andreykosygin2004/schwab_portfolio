import dash
from dash import html, dcc, Input, Output, callback, dash_table
import numpy as np
import pandas as pd
import plotly.express as px

from analytics.holdings_intel import (
    concentration_metrics,
    covariance_matrix,
    risk_contributions,
    scenario_impact,
)
from analytics.risk import compute_returns
from analytics.constants import DEFAULT_START_DATE_ANALYSIS
from analytics.portfolio import load_holdings_timeseries, load_portfolio_series
from viz.plots import empty_figure

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
    weights = weights[weights > 0]
    if weights.empty:
        empty = empty_figure("No holdings available.")
        return "No holdings available.", None, empty, [], "No holdings available.", "No holdings available."

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

    method_note = "Method: snapshot weights (current holdings only)."
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
