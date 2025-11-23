import dash
from dash import Dash, html, dcc
import pandas as pd
import plotly.express as px


# Load data
transactions = pd.read_csv("data/schwab_transactions.csv", parse_dates=["Date"])
prices = pd.read_csv("data/historical_prices.csv", index_col=0, parse_dates=True)
benchmarks = pd.read_csv("data/benchmark_prices.csv", index_col=0, parse_dates=True)
vol = pd.read_csv("data/vol.csv", index_col=0, parse_dates=True)
treasury = pd.read_csv("data/treasury.csv", index_col=0, parse_dates=True)
cpi = pd.read_csv("data/cpi.csv", index_col=0, parse_dates=True)


# Calculate porfolio value over time
portfolio_value = prices.sum(axis=1)
print(portfolio_value.head())

portfolio_fig = px.line(portfolio_value, title="Portfolio Value Over Time",
                        labels={"index": "Date", "value": "Portfolio Value ($)"})


# Calculate benchmarks
bench_fig = px.line(benchmarks, title="Portfolio Benchmarks",
                    labels={"index": "Date", "value": "Price"})


# Calculate macro values
macro_df = pd.concat([vol, treasury, cpi], axis=1)
macro_df.columns = ["VIX", "10Y Treasury", "CPI"]

macro_fig = px.line(macro_df, title="Macro Indicators Over Time",
                    labels={"index": "Date", "value": "Value"})


# Initialize app
app = dash.Dash(__name__)
app.title = "Schwab Portfolio Dashboard"

app.layout = html.Div([
    html.H1("Schwab Portfolio Dashboard", style={"textAlign": "center"}),

    html.H2("Portfolio Holdings Over Time"),
    dcc.Graph(id="portfolio-graph", figure = portfolio_fig),

    html.H2("Benchmarks Comparison"),
    dcc.Graph(id="benchmark-graph", figure = bench_fig),

    html.H2("Macro Overlays"),
    dcc.Graph(id="macro-graph", figure = macro_fig)
])


# Run app
if __name__ == "__main__":
    app.run(debug=True)