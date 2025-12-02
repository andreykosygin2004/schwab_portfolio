import dash
from dash import Dash, html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px

dash.register_page(__name__, path="/macro", name="Macro Analysis")

vol = pd.read_csv("data/vol.csv", parse_dates=['Date'], index_col='Date')
treasury = pd.read_csv("data/treasury.csv", parse_dates=['DATE'], index_col='DATE')
cpi = pd.read_csv("data/cpi.csv", parse_dates=['DATE'], index_col='DATE')

macro_df = pd.concat([vol, treasury, cpi], axis=1)

layout = html.Div([
    html.H2("Macro Overview"),

    html.Label("Selected macros:"),
    dcc.Dropdown(
        id="macro-select",
        options=[{"label": t, "value": t} for t in macro_df.columns],
        value=[],
        multi=True
    ),

    html.Br(),
    html.Br(),
    html.Label("Select Date Range:"),
    dcc.DatePickerRange(id="date-range", min_date_allowed=macro_df.index.min(),
                        max_date_allowed=macro_df.index.max(), start_date=macro_df.index.min(),
                        end_date=macro_df.index.max()),

    html.Br(), html.Br(),

    dcc.Graph(id="macro-graph")
])

@callback(
    Output("macro-graph", "figure"),
    Input("macro-select", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date")
)
def update_macro_graph(selected_macros, start_date, end_date):
    if not selected_macros:
        return px.line(title="No macros selected")
    
    df = macro_df.loc[start_date:end_date, selected_macros]

    fig = px.line(df, title="Selected Macros")
    fig.update_layout(legend_title_text="Macros")

    return fig