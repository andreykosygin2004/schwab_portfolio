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
    make_entry_event_label,
    make_weekly_regimes,
    make_weekly_returns,
    predict_proba,
    resolve_splits,
    split_by_time,
    train_model,
)
from analytics.constants import ANALYSIS_END, ANALYSIS_START
from viz.plots import empty_figure

dash.register_page(__name__, path="/regime-forecast", name="Regime Forecaster")

HORIZONS = [2, 3, 4]
MODELS = ["Logistic Regression", "Gradient Boosting"]
TARGETS = [
    "Enter Risk-Off / Credit Stress",
    "Enter Rates Shock",
    "Enter Neutral / Compression",
]
TARGET_LABELS = {
    "Enter Risk-Off / Credit Stress": "Risk-Off / Credit Stress",
    "Enter Rates Shock": "Rates Shock",
    "Enter Neutral / Compression": "Neutral / Transition",
}


layout = html.Div([
    html.Br(),
    html.H2("Regime Forecast"),
    html.P("Weekly probability of entering the selected regime within the chosen horizon."),

    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Horizon (weeks)"),
                    dcc.Dropdown(
                        id="forecast-horizon",
                        options=[{"label": str(h), "value": h} for h in HORIZONS],
                        value=3,
                        clearable=False,
                        className="forecast-dropdown",
                    ),
                ], xs=12, sm=6, md=3),
                dbc.Col([
                    html.Label("Model"),
                    dcc.Dropdown(
                        id="forecast-model",
                        options=[{"label": m, "value": m} for m in MODELS],
                        value="Logistic Regression",
                        clearable=False,
                        className="forecast-dropdown",
                    ),
                ], xs=12, sm=6, md=3),
                dbc.Col([
                    html.Label("Target"),
                    dcc.Dropdown(
                        id="forecast-target",
                        options=[{"label": t, "value": t} for t in TARGETS],
                        value="Enter Risk-Off / Credit Stress",
                        clearable=False,
                        className="forecast-dropdown",
                    ),
                ], xs=12, sm=6, md=4),
                dbc.Col([
                    html.Label("Data"),
                    dbc.Button(
                        "Hard refresh proxies",
                        id="forecast-refresh",
                        size="sm",
                        color="secondary",
                        style={"marginTop": "6px"},
                    ),
                ], xs=12, sm=6, md=2),
            ], className="g-3"),
        ]),
        style={"borderRadius": "10px", "border": "1px solid #e6e9ee"},
    ),

    html.Br(),
    html.Div(id="forecast-warning"),
    html.Div(id="forecast-current"),
    html.Br(),
    dcc.Loading(dcc.Graph(id="forecast-prob")),
    html.Br(),
    html.Div(id="forecast-latest-title"),
    html.P(
        "These are separate one-vs-rest event probabilities; they do not sum to 1 because events can overlap."
    ),
    html.Div(id="forecast-latest-probs"),
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
    Output("forecast-latest-title", "children"),
    Output("forecast-latest-probs", "children"),
    Input("forecast-horizon", "value"),
    Input("forecast-model", "value"),
    Input("forecast-target", "value"),
    Input("forecast-refresh", "n_clicks"),
)
def update_forecast(horizon, model_name, target_choice, refresh_clicks):
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
        return empty, empty, "No data available.", None, None, None, None

    end = latest
    weekly_ret = make_weekly_returns(start, end)
    regimes = make_weekly_regimes(start, end)
    if weekly_ret.empty or regimes.empty:
        empty = empty_figure("No data available.")
        return empty, empty, "No data available.", None, None, None, None

    target_label = TARGET_LABELS.get(target_choice, "Risk-Off / Credit Stress")
    try:
        y = make_entry_event_label(regimes, target_label, horizon)
    except ValueError as exc:
        empty = empty_figure(str(exc))
        return empty, empty, str(exc), None, None, None, None
    X_full = build_features(weekly_ret, regimes)
    if X_full.empty or y.empty:
        empty = empty_figure("No data available.")
        return empty, empty, "No data available.", None, None, None, None

    aligned = X_full.index.intersection(y.index)
    X = X_full.reindex(aligned).dropna()
    y = y.reindex(X.index)

    split_base = X.loc[X.index <= ANALYSIS_END]
    y_split = y.reindex(split_base.index)
    splits = resolve_splits(split_base.index, default_splits())
    split = split_by_time(split_base, y_split, splits)
    if split["X_train"].empty or split["X_test"].empty:
        empty = empty_figure("Insufficient data for splits.")
        return empty, empty, "Insufficient data for splits.", None, None, None, None

    model, scaler = train_model(split["X_train"], split["y_train"], model_name)
    probs = predict_proba(model, scaler, X_full.dropna())
    probs = probs.sort_index()

    eval_idx = probs.index.intersection(y.index)
    eval_idx = eval_idx[(eval_idx >= ANALYSIS_START) & (eval_idx <= ANALYSIS_END)]
    test_probs = probs.loc[eval_idx]
    test_labels = y.loc[eval_idx]
    metrics = evaluate_model(test_labels, test_probs)

    prob_fig = go.Figure()
    prob_fig.add_trace(go.Scatter(x=probs.index, y=probs.values, name=f"P(Enter {target_label})"))
    regime_hits = regimes.reindex(probs.index)
    hit_idx = regime_hits[regime_hits == target_label].index
    if not hit_idx.empty:
        runs = []
        run_start = hit_idx[0]
        prev = hit_idx[0]
        for dt in hit_idx[1:]:
            if (dt - prev).days <= 7:
                prev = dt
                continue
            runs.append((run_start, prev))
            run_start = dt
            prev = dt
        runs.append((run_start, prev))
        for start_dt, end_dt in runs:
            prob_fig.add_vrect(
                x0=start_dt,
                x1=end_dt,
                fillcolor="rgba(220, 38, 38, 0.08)",
                line_width=0,
                layer="below",
            )
    prob_fig.add_vrect(x0=ANALYSIS_START, x1=ANALYSIS_END, fillcolor="LightSalmon", opacity=0.2, line_width=0)
    prob_fig.update_layout(title=f"P(Enter {target_label} in next {horizon}w)", height=420)
    prob_fig.update_yaxes(title_text="Probability", tickformat=".0%")

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
            html.Div(f"Current {target_label} Probability (next {horizon}w): {current_prob:.1%}"),
            html.Div(f"As of: {current_date}"),
            html.Div(f"Horizon: {horizon} weeks"),
        ]),
        style={"maxWidth": "420px"},
    )

    warning = None
    if latest < (today - pd.Timedelta(days=7)):
        warning = dbc.Alert("Data not up to date. Click Refresh.", color="warning")

    latest_title = html.Small(f"Latest probabilities ({model_name}, {horizon}w horizon)")

    latest_rows = []
    for choice, label in TARGET_LABELS.items():
        try:
            y_target = make_entry_event_label(regimes, label, horizon)
        except ValueError:
            latest_rows.append({"Target": choice, "Probability": "N/A"})
            continue
        aligned_target = X_full.index.intersection(y_target.index)
        X_target = X_full.reindex(aligned_target).dropna()
        y_target = y_target.reindex(X_target.index)
        if X_target.empty:
            latest_rows.append({"Target": choice, "Probability": "N/A"})
            continue
        split_target = split_by_time(X_target, y_target, splits)
        if split_target["X_train"].empty:
            latest_rows.append({"Target": choice, "Probability": "N/A"})
            continue
        model_t, scaler_t = train_model(split_target["X_train"], split_target["y_train"], model_name)
        probs_t = predict_proba(model_t, scaler_t, X_target)
        latest_rows.append({"Target": choice, "Probability": f"{probs_t.iloc[-1]:.1%}"})

    latest_table = dbc.Table.from_dataframe(
        pd.DataFrame(latest_rows),
        striped=True,
        bordered=True,
        hover=True,
        size="sm",
    )

    return prob_fig, calib_fig, metrics_text, current_card, warning, latest_title, latest_table
