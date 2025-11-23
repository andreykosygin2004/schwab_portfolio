import dash
from dash import Dash, html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px

dash.register_page(__name__, path="/", name="Home")

prices = pd.read_csv("data/historical_prices.csv", index_col=0, parse_dates=True)
tickers = prices.columns.tolist()

layout = html.Div([
    html.Br(),
    html.H2("Portfolio Overview"),

    html.Label("Select Tickers:"),
    dcc.Dropdown(
        id="ticker-select",
        options=[{"label": t, "value": t} for t in tickers],
        value=[],
        multi=True
    ),

    html.Br(),
    html.Label("Select Date Range:"),
    dcc.DatePickerRange(id="date-range", min_date_allowed=prices.index.min(),
                        max_date_allowed=prices.index.max(), start_date=prices.index.min(),
                        end_date=prices.index.max()),

    html.Br(), html.Br(),

    dcc.Graph(id="ticker-graph")
])

@callback(
    Output("ticker-graph", "figure"),
    Input("ticker-select", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date")
)
def update_ticker_graph(selected_tickers, start_date, end_date):
    if not selected_tickers:
        return px.line(title="No tickers selected")
    
    df = prices.loc[start_date:end_date, selected_tickers]

    fig = px.line(df, title="Selected Tickers")
    fig.update_layout(legend_title_text="Tickers")

    return fig