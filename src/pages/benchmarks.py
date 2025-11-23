import dash
from dash import Dash, html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px

dash.register_page(__name__, path="/benchmarks", name="Benchmark Analysis")

benchmarks_prices = pd.read_csv("data/benchmark_prices.csv", index_col=0, parse_dates=True)
benchmarks = benchmarks_prices.columns.tolist()

layout = html.Div([
    html.H2("Benchmark Overview"),

    html.Label("Selected benchmarks:"),
    dcc.Dropdown(
        id="benchmark-select",
        options=[{"label": t, "value": t} for t in benchmarks],
        value=[],
        multi=True
    ),

    html.Br(),
    html.Br(),
    html.Label("Select Date Range:"),
    dcc.DatePickerRange(id="date-range", min_date_allowed=benchmarks_prices.index.min(),
                        max_date_allowed=benchmarks_prices.index.max(), start_date=benchmarks_prices.index.min(),
                        end_date=benchmarks_prices.index.max()),

    html.Br(), html.Br(),

    dcc.Graph(id="benchmark-graph")
])

@callback(
    Output("benchmark-graph", "figure"),
    Input("benchmark-select", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date")
)
def update_benchmark_graph(selected_benchmarks, start_date, end_date):
    if not selected_benchmarks:
        return px.line(title="No benchmarks selected")
    
    df = benchmarks_prices.loc[start_date:end_date, selected_benchmarks]

    fig = px.line(df, title="Selected Benchmarks")
    fig.update_layout(legend_title_text="Benchmarks")

    return fig