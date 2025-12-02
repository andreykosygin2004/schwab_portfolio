import dash
from dash import Dash, html, dcc, Input, Output, callback, dash_table
import pandas as pd
import plotly.express as px

dash.register_page(__name__, path="/", name="Home")

prices = pd.read_csv("data/historical_prices.csv", index_col=0, parse_dates=True)
tickers = prices.columns.tolist()

holdings_ts = pd.read_csv("data/holdings_timeseries.csv", index_col=0, parse_dates=True)
summary = pd.read_csv("data/portfolio_summary.csv", header=None, index_col=0)
summary.columns = ["value"]

mv_tickers = [c for c in holdings_ts.columns if c.startswith("MV_")]
mv_tickers_clean = [c.replace("MV_", "") for c in mv_tickers]

transactions = pd.read_csv("data/schwab_transactions.csv", parse_dates=["Date"])
transactions["ActionType"] = transactions.apply(
    lambda r: r["Action"] if pd.notna(r["Action"]) else "OTHER", axis=1
)

layout = html.Div([
    html.Br(),
    html.Br(),
    html.H2("Portfolio Overview"),

    html.Br(),
    html.H3("Holdings Market Values & Portfolio Total"),
    html.Label("Select Holdings:"),
    dcc.Dropdown(
        id="mv-ticker-select",
        options=[{"label": t, "value": t} for t in mv_tickers_clean] + [{"label": "Portfolio Total", "value": "Portfolio Total"}],
        value=[],
        multi=True
    ),
    html.Label("Select Date Range:"),
    dcc.DatePickerRange(
        id="mv-date-range",
        min_date_allowed=holdings_ts.index.min(),
        max_date_allowed=holdings_ts.index.max(),
        start_date=holdings_ts.index.min(),
        end_date=holdings_ts.index.max()
    ),
    dcc.Graph(id="mv-graph"),

    html.Br(),
    html.H3("Portfolio Summary"),
    
    dash_table.DataTable(
        id="mv-summary-table",
        columns=[
            {"name": "Metric", "id": "Metric"},
            {"name": "Value", "id": "Value"}
        ],
        data=[],  # filled by callback
        style_cell={"textAlign": "left", "padding": "8px"},
        style_header={"fontWeight": "bold"},
        style_table={"width": "60%", "margin": "auto"}
    ),

    html.Hr(),

    html.H3("Portfolio Total (Excluding Negative MoneyLink Transfers)"),
    html.Label("Select Date Range:"),
    dcc.DatePickerRange(
        id="pv-clean-date-range",
        min_date_allowed=holdings_ts.index.min(),
        max_date_allowed=holdings_ts.index.max(),
        start_date=holdings_ts.index.min(),
        end_date=holdings_ts.index.max()
    ),
    dcc.Graph(id="pv-clean-graph"),

    
    html.Hr(),

    html.Br(),
    html.H3("General Overview (Price History)"),

    html.Label("Select Tickers:"),
    dcc.Dropdown(
        id="ticker-select",
        options=[{"label": t, "value": t} for t in tickers],
        value=[],
        multi=True
    ),

    html.Br(),
    html.Label("Select Date Range:"),
    dcc.DatePickerRange(id="date-range",
                        min_date_allowed=prices.index.min(),
                        max_date_allowed=prices.index.max(),
                        start_date=prices.index.min(),
                        end_date=prices.index.max()
    ),

    html.Br(), html.Br(),

    dcc.Graph(id="ticker-graph")
])


@callback(
        Output("mv-graph", "figure"),
        Input("mv-ticker-select", "value"),
        Input("mv-date-range", "start_date"),
        Input("mv-date-range", "end_date")
)
def update_mv_graph(selected_holdings, start_date, end_date):
    if not selected_holdings:
        return px.line(title="No holdings selected")
    
    mv_cols = [f"MV_{t}" for t in selected_holdings if f"MV_{t}" in holdings_ts.columns]
    
    show_portfolio_total = "Portfolio Total" in selected_holdings
    if show_portfolio_total:
        df = holdings_ts.loc[start_date:end_date, [c for c in holdings_ts.columns if c.startswith("MV_")]].copy()
        df = df.sum(axis=1).to_frame(name="Portfolio Total")
    else:
        if not mv_cols:
            return px.line(title="No valid tickers selected")
        df = holdings_ts.loc[start_date:end_date, mv_cols].copy()

    fig = px.line(df, title="Market Values")
    fig.update_layout(legend_title_text="Tickers")

    return fig


@callback(
    Output("mv-summary-table", "data"),
    Input("mv-graph", "figure")  # triggers refresh when graph updates
)
def update_portfolio_summary_table(_):
    df = summary.copy()

    # Convert to rows for DataTable
    table_data = [
        {"Metric": idx, "Value": f"{val:.4f}" if isinstance(val, (float, int)) else str(val)}
        for idx, val in df["value"].items()
    ]

    return table_data


@callback(
    Output("pv-clean-graph", "figure"),
    Input("pv-clean-date-range", "start_date"),
    Input("pv-clean-date-range", "end_date")
)
def update_pv_clean_graph(start_date, end_date):
    df = holdings_ts.copy()
    df = df.loc[start_date:end_date]

    tx = transactions.copy()

    # Parse Date
    tx["Date"] = pd.to_datetime(tx["Date"], errors="coerce")

    # Parse Amount: remove $, commas, parentheses, minus signs
    tx["Amount"] = (
        tx["Amount"]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
    )
    tx["Amount"] = pd.to_numeric(tx["Amount"], errors="coerce").fillna(0)

    # Ensure ActionType exists (your classify_action logic)
    if "ActionType" not in tx.columns:
        from analytics import classify_action
        tx["ActionType"] = tx.apply(
            lambda r: classify_action(r.get("Action", ""), r.get("Description", "")),
            axis=1
        )

    # -------------------------------
    # 3. Remove negative MoneyLink Transfers
    # -------------------------------
    neg_ml_mask = (tx["ActionType"] == "MoneyLink Transfer") & (tx["Amount"] < 0)
    tx.loc[neg_ml_mask, "Amount"] = 0  # remove negative transfers

    # -------------------------------
    # 4. Recompute clean cash balance
    # -------------------------------
    cash_balance_clean = tx.groupby("Date")["Amount"].sum().cumsum()
    cash_balance_clean = cash_balance_clean.reindex(holdings_ts.index).ffill().fillna(0)

    # -------------------------------
    # 5. Compute Portfolio Total Clean
    # -------------------------------
    mv_cols = [c for c in holdings_ts.columns if c.startswith("MV_")]

    df_clean = holdings_ts.loc[start_date:end_date, mv_cols].copy()
    df_clean["Portfolio Total Clean"] = df_clean.sum(axis=1) + cash_balance_clean.loc[start_date:end_date]

    # -------------------------------
    # 6. Create figure
    # -------------------------------
    fig = px.line(
        df_clean,
        y="Portfolio Total Clean",
        title="Portfolio Value (Negative MoneyLink Transfers Removed)"
    )

    fig.update_layout(
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Date",
        height=450
    )

    return fig


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