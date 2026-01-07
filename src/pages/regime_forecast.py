import dash
from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analytics.regime_probability import (
    build_features,
    default_splits,
    evaluate_model,
    get_latest_available_date,
    make_labels,
    make_weekly_regimes,
    make_weekly_returns,
    predict_proba,
    resolve_splits,
    split_by_time,
    train_model,
)
from analytics.constants import ANALYSIS_END, ANALYSIS_START
from viz.plots import empty_figure

dash.register_page(__name__, path="/regime-forecast", name="Regime Forecast")

HORIZONS = [2, 3, 4]
MODELS = ["Logistic Regression", "Gradient Boosting"]


layout = html.Div([
    html.Br(),
    html.H2("Regime Forecast"),
    html.P("Weekly probability of entering Risk-Off / Credit Stress within the chosen horizon."),

    html.Div([
        html.Div([
            html.Label("Horizon (weeks)"),
            dcc.Dropdown(
                id="forecast-horizon",
                options=[{"label": str(h), "value": h} for h in HORIZONS],
                value=3,
                clearable=False,
            ),
        ], style={"maxWidth": "200px"}),
        html.Div([
            html.Label("Model"),
            dcc.Dropdown(
                id="forecast-model",
                options=[{"label": m, "value": m} for m in MODELS],
                value="Logistic Regression",
                clearable=False,
            ),
        ], style={"maxWidth": "260px"}),
        html.Div([
            html.Label("Mode"),
            dcc.RadioItems(
                id="forecast-mode",
                options=[
                    {"label": "Live", "value": "Live"},
                    {"label": "Evaluation-only", "value": "Eval"},
                ],
                value="Live",
                inline=True,
            ),
        ], style={"maxWidth": "320px"}),
        html.Div([
            html.Label("Data"),
            dbc.Button("Hard refresh proxies", id="forecast-refresh", size="sm", color="secondary"),
        ]),
    ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap"}),

    html.Br(),
    html.Div(id="forecast-warning"),
    html.Div(id="forecast-current"),
    html.Br(),
    dcc.Loading(dcc.Graph(id="forecast-prob")),
    html.Br(),
    dcc.Loading(dcc.Graph(id="forecast-calibration")),
    html.Br(),
    html.Div(id="forecast-metrics"),
])


@callback(
    Output("forecast-prob", "figure"),
    Output("forecast-calibration", "figure"),
    Output("forecast-metrics", "children"),
    Output("forecast-current", "children"),
    Output("forecast-warning", "children"),
    Input("forecast-horizon", "value"),
    Input("forecast-model", "value"),
    Input("forecast-mode", "value"),
    Input("forecast-refresh", "n_clicks"),
)
def update_forecast(horizon, model_name, mode, refresh_clicks):
    start = pd.Timestamp("2005-01-01")
    today = pd.Timestamp.today().normalize()
    # Live mode uses the latest available proxy date; evaluation stays anchored to the test window.
    latest = get_latest_available_date(
        start,
        today,
        refresh=bool(refresh_clicks),
        hard_refresh=bool(refresh_clicks),
    )
    if latest is None:
        empty = empty_figure("No data available.")
        return empty, empty, "No data available.", None, None

    end = latest if mode == "Live" else ANALYSIS_END
    weekly_ret = make_weekly_returns(start, end)
    regimes = make_weekly_regimes(start, end)
    if weekly_ret.empty or regimes.empty:
        empty = empty_figure("No data available.")
        return empty, empty, "No data available.", None, None

    y = make_labels(regimes, horizon)
    X_full = build_features(weekly_ret, regimes)
    if X_full.empty or y.empty:
        empty = empty_figure("No data available.")
        return empty, empty, "No data available.", None, None

    aligned = X_full.index.intersection(y.index)
    X = X_full.reindex(aligned).dropna()
    y = y.reindex(X.index)

    split_base = X.loc[X.index <= ANALYSIS_END]
    y_split = y.reindex(split_base.index)
    splits = resolve_splits(split_base.index, default_splits())
    split = split_by_time(split_base, y_split, splits)
    if split["X_train"].empty or split["X_test"].empty:
        empty = empty_figure("Insufficient data for splits.")
        return empty, empty, "Insufficient data for splits.", None, None

    model, scaler = train_model(split["X_train"], split["y_train"], model_name)
    probs = predict_proba(model, scaler, X_full.dropna())
    probs = probs.sort_index()

    eval_idx = probs.index.intersection(y.index)
    eval_idx = eval_idx[(eval_idx >= ANALYSIS_START) & (eval_idx <= ANALYSIS_END)]
    test_probs = probs.loc[eval_idx]
    test_labels = y.loc[eval_idx]
    metrics = evaluate_model(test_labels, test_probs)

    prob_fig = go.Figure()
    prob_fig.add_trace(go.Scatter(x=probs.index, y=probs.values, name="P(Risk-Off in horizon)"))
    prob_fig.add_vrect(x0=ANALYSIS_START, x1=ANALYSIS_END, fillcolor="LightSalmon", opacity=0.2, line_width=0)
    prob_fig.update_layout(title="Risk-Off Probability (Weekly)", height=420)
    prob_fig.update_yaxes(title_text="Probability", tickformat=".0%")
    if mode == "Eval":
        prob_fig.update_xaxes(range=[ANALYSIS_START, ANALYSIS_END])

    calib_fig = empty_figure("No calibration data.")
    if test_labels.nunique() > 1:
        bins = pd.qcut(test_probs, q=10, duplicates="drop")
        calib = pd.DataFrame({"y": test_labels, "p": test_probs, "bin": bins}).groupby("bin").mean()
        calib_fig = go.Figure()
        calib_fig.add_trace(go.Scatter(x=calib["p"], y=calib["y"], mode="markers+lines", name="Calibration"))
        calib_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line={"dash": "dash"}))
        calib_fig.update_layout(title="Calibration (Test Window)", height=320)
        calib_fig.update_xaxes(title="Predicted Probability")
        calib_fig.update_yaxes(title="Observed Frequency")

    metrics_text = (
        f"Prevalence: {metrics.get('prevalence', np.nan):.2%} | "
        f"ROC-AUC: {metrics.get('roc_auc', np.nan):.3f} | "
        f"PR-AUC: {metrics.get('pr_auc', np.nan):.3f} | "
        f"Brier: {metrics.get('brier', np.nan):.4f}"
    )
    current_prob = probs.iloc[-1]
    current_date = probs.index[-1].date()
    current_card = dbc.Card(
        dbc.CardBody([
            html.Div(f"Current Risk-Off Probability (next {horizon}w): {current_prob:.1%}"),
            html.Div(f"As of: {current_date}"),
            html.Div(f"Horizon: {horizon} weeks"),
        ]),
        style={"maxWidth": "420px"},
    )

    warning = None
    if mode == "Live" and latest < (today - pd.Timedelta(days=7)):
        warning = dbc.Alert("Data not up to date. Click Refresh.", color="warning")

    return prob_fig, calib_fig, metrics_text, current_card, warning
