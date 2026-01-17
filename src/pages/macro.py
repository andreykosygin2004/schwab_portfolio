import dash
from dash import html, dcc, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from analytics_macro import (
    compute_returns,
    rolling_vol,
    rolling_corr,
    rolling_beta,
    beta_and_r2,
    max_drawdown,
    normalize_to_100,
    load_portfolio_series,
    load_fred_series,
    load_ticker_prices,
)
from analytics.portfolio import risk_free_warning
from analytics.constants import DEFAULT_START_DATE_ANALYSIS

dash.register_page(__name__, path="/macro", name="Macro Analysis")

PORTFOLIO_SERIES = load_portfolio_series()
PORTFOLIO_MIN = PORTFOLIO_SERIES.index.min()
PORTFOLIO_MAX = PORTFOLIO_SERIES.index.max()
DEFAULT_END = PORTFOLIO_MAX
DEFAULT_START = max(PORTFOLIO_MIN, DEFAULT_START_DATE_ANALYSIS)

PROXY_OPTIONS = ["TLT", "TIP", "GLD", "USO", "HYG", "LQD", "UUP"]
DEFAULT_PROXIES = ["TLT", "TIP", "GLD", "USO", "HYG", "LQD"]
BENCHMARK_OPTIONS = ["SPY", "QQQ"]
FREQ_OPTIONS = ["Daily", "Weekly"]

CARD_STYLE = {
    "border": "1px solid #e0e0e0",
    "borderRadius": "10px",
    "padding": "12px 14px",
    "boxShadow": "0 1px 2px rgba(0, 0, 0, 0.08)",
    "backgroundColor": "#ffffff",
}
GRID_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
    "gap": "16px",
}
GRAPH_STYLE = {"height": "420px", "maxHeight": "420px"}
SPARK_STYLE = {"height": "140px", "maxHeight": "140px"}
INFO_STYLE = {"cursor": "pointer", "textDecoration": "underline"}


def _empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, height=260, autosize=False)
    return fig


def _sparkline(series: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    if not series.empty:
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", line={"width": 2}))
    fig.update_layout(
        title=title,
        height=140,
        autosize=False,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _format_pct(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value * 100:.2f}%"


def _period_change(series: pd.Series, months: int) -> float:
    if series.empty:
        return np.nan
    end_date = series.index.max()
    start_date = end_date - pd.DateOffset(months=months)
    prior = series.loc[:start_date]
    if prior.empty:
        return np.nan
    return (series.iloc[-1] / prior.iloc[-1]) - 1.0


def _ytd_change(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    end_date = series.index.max()
    start_date = pd.Timestamp(year=end_date.year, month=1, day=1)
    prior = series.loc[:start_date]
    if prior.empty:
        return np.nan
    return (series.iloc[-1] / prior.iloc[-1]) - 1.0


layout = html.Div([
    html.Br(),
    html.Br(),
    html.H2("Macro Dashboard"),
    html.P("Monitor macro drivers and proxy signals that influence portfolio risk and returns."),
    html.Div(
        risk_free_warning(),
        style={"color": "#b45309", "marginBottom": "8px"},
    ) if risk_free_warning() else html.Div(),
    html.Br(),
    html.Br(),

    html.Div([
        html.Div([
            html.Label("Date range"),
            dcc.DatePickerRange(
                id="macro-date-range",
                min_date_allowed=PORTFOLIO_MIN,
                max_date_allowed=PORTFOLIO_MAX,
                start_date=DEFAULT_START.date(),
                end_date=DEFAULT_END.date(),
            ),
        ], style={"maxWidth": "360px"}),
        html.Div([
            html.Label("Frequency"),
            dcc.RadioItems(
                id="macro-frequency",
                options=[{"label": f, "value": f} for f in FREQ_OPTIONS],
                value="Daily",
                inline=True,
            ),
        ], style={"maxWidth": "260px"}),
        html.Div([
            html.Label("Benchmark"),
            dcc.Dropdown(
                id="macro-benchmark",
                options=[{"label": b, "value": b} for b in BENCHMARK_OPTIONS],
                value="SPY",
                clearable=False,
            ),
        ], style={"maxWidth": "220px"}),
        html.Div([
            html.Label("Macro proxies"),
            dcc.Dropdown(
                id="macro-proxies",
                options=[{"label": t, "value": t} for t in PROXY_OPTIONS],
                value=DEFAULT_PROXIES,
                multi=True,
            ),
        ], style={"minWidth": "320px"}),
    ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap"}),

    html.Br(),
    html.Hr(),
    html.Br(),
    html.H3("Macro Regime Overview"),
    html.Br(),
    html.Div([
        html.Div([
            html.Div(id="macro-vol-metric", style={"fontSize": "20px", "fontWeight": "600"}),
            dcc.Loading(dcc.Graph(id="macro-vol-spark", config={"displayModeBar": False, "responsive": False}, style=SPARK_STYLE)),
        ], style=CARD_STYLE),
        html.Div([
            html.Div(id="macro-corr-metric", style={"fontSize": "20px", "fontWeight": "600"}),
            dcc.Loading(dcc.Graph(id="macro-corr-spark", config={"displayModeBar": False, "responsive": False}, style=SPARK_STYLE)),
        ], style=CARD_STYLE),
        html.Div([
            html.Div(id="macro-beta-metric", style={"fontSize": "20px", "fontWeight": "600"}),
            dcc.Loading(dcc.Graph(id="macro-beta-spark", config={"displayModeBar": False, "responsive": False}, style=SPARK_STYLE)),
        ], style=CARD_STYLE),
        html.Div([
            html.Div(id="macro-dd-metric", style={"fontSize": "20px", "fontWeight": "600"}),
            dcc.Loading(dcc.Graph(id="macro-dd-spark", config={"displayModeBar": False, "responsive": False}, style=SPARK_STYLE)),
        ], style=CARD_STYLE),
    ], style=GRID_STYLE),

    html.Br(),
    html.Hr(),
    html.Br(),
    html.H3("Macro Exposure Summary", style={"textAlign": "center"}),
    html.Br(),
    dash_table.DataTable(
        id="macro-exposure-table",
        columns=[
            {"name": "Proxy", "id": "proxy"},
            {"name": "Correlation", "id": "corr"},
            {"name": "Beta", "id": "beta"},
            {"name": "R^2", "id": "r2"},
            {"name": "Proxy 3M Return", "id": "ret_3m"},
            {"name": "Proxy Ann. Vol", "id": "ann_vol"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px"},
    ),

    html.Br(),
    html.Hr(),
    html.Br(),
    html.H3("Rates / Inflation / Credit"),
    html.Br(),
    html.P(
        "Normalized indices show how rates, inflation, and credit proxies have moved together. "
        "The table highlights recent momentum."
    ),
    dcc.Loading(dcc.Graph(id="macro-rates-chart", style=GRAPH_STYLE, config={"responsive": False})),
    dash_table.DataTable(
        id="macro-rates-table",
        columns=[
            {"name": "Series", "id": "series"},
            {"name": "Last", "id": "last"},
            {"name": "1M Change", "id": "change_1m"},
            {"name": "3M Change", "id": "change_3m"},
            {"name": "YTD Change", "id": "change_ytd"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px"},
    ),

    html.Br(),
    html.Hr(),
    html.Br(),
    html.H3("Commodities / USD / Risk Sentiment"),
    html.Br(),
    html.P(
        "Correlations show how the portfolio tracks macro proxies and broad risk sentiment."
    ),
    dcc.Loading(dcc.Graph(id="macro-heatmap", style=GRAPH_STYLE, config={"responsive": False})),
])



@callback(
    Output("macro-vol-spark", "figure"),
    Output("macro-corr-spark", "figure"),
    Output("macro-beta-spark", "figure"),
    Output("macro-dd-spark", "figure"),
    Output("macro-vol-metric", "children"),
    Output("macro-corr-metric", "children"),
    Output("macro-beta-metric", "children"),
    Output("macro-dd-metric", "children"),
    Output("macro-rates-chart", "figure"),
    Output("macro-rates-table", "data"),
    Output("macro-heatmap", "figure"),
    Output("macro-exposure-table", "data"),
    Input("macro-date-range", "start_date"),
    Input("macro-date-range", "end_date"),
    Input("macro-frequency", "value"),
    Input("macro-benchmark", "value"),
    Input("macro-proxies", "value"),
    Input("portfolio-selector", "value"),
)
def update_macro_dashboard(start_date, end_date, freq, benchmark, proxies, portfolio_id):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    proxies = proxies or []

    portfolio_series = load_portfolio_series(portfolio_id=portfolio_id or "schwab")
    portfolio_prices = portfolio_series.loc[start:end].dropna()
    if portfolio_prices.empty:
        empty = _empty_fig("No data available for selected range.")
        return (
            empty, empty, empty, empty,
            "Rolling vol: n/a", "Rolling corr: n/a", "Rolling beta: n/a", "Max drawdown: n/a",
            empty, [], empty, [],
        )

    benchmark_prices = load_ticker_prices([benchmark], start=start, end=end)
    if benchmark_prices.empty or benchmark not in benchmark_prices.columns:
        print(f"[WARN] Missing benchmark data for {benchmark}.")
        benchmark_prices = pd.DataFrame(index=portfolio_prices.index)

    proxy_prices = load_ticker_prices(proxies, start=start, end=end)

    portfolio_prices_chart = portfolio_prices
    benchmark_prices_chart = benchmark_prices
    proxy_prices_chart = proxy_prices
    if freq == "Weekly":
        portfolio_prices_chart = portfolio_prices.resample("W-FRI").last()
        if not benchmark_prices.empty:
            benchmark_prices_chart = benchmark_prices.resample("W-FRI").last()
        if not proxy_prices.empty:
            proxy_prices_chart = proxy_prices.resample("W-FRI").last()

    port_ret = compute_returns(portfolio_prices, freq)
    bench_ret = compute_returns(benchmark_prices[benchmark], freq) if benchmark_prices is not None and not benchmark_prices.empty else pd.Series(dtype=float)

    window = 30 if freq == "Daily" else 12
    ann_factor = 252 if freq == "Daily" else 52

    vol_series = rolling_vol(port_ret, window) * np.sqrt(ann_factor)
    corr_series = rolling_corr(port_ret, bench_ret, window)
    beta_series = rolling_beta(port_ret, bench_ret, window)

    vol_metric = f"Rolling vol ({window}): {_format_pct(vol_series.dropna().iloc[-1])}" if not vol_series.empty else "Rolling vol: n/a"
    corr_metric = f"Rolling corr ({window}): {corr_series.dropna().iloc[-1]:.2f}" if not corr_series.empty else "Rolling corr: n/a"
    beta_metric = f"Rolling beta ({window}): {beta_series.dropna().iloc[-1]:.2f}" if not beta_series.empty else "Rolling beta: n/a"

    port_dd, port_dd_series = max_drawdown(portfolio_prices_chart)
    if benchmark_prices_chart is not None and not benchmark_prices_chart.empty:
        bench_dd, bench_dd_series = max_drawdown(benchmark_prices_chart[benchmark].dropna())
    else:
        bench_dd, bench_dd_series = np.nan, pd.Series(dtype=float)
    dd_metric = f"Max DD: Port {_format_pct(port_dd)} | Bench {_format_pct(bench_dd)}"

    vol_fig = _sparkline(vol_series, "Rolling Volatility")
    corr_fig = _sparkline(corr_series, "Rolling Correlation")
    beta_fig = _sparkline(beta_series, "Rolling Beta")
    dd_fig = _sparkline(pd.concat([port_dd_series, bench_dd_series], axis=1).min(axis=1), "Drawdown")

    # Section B: rates, inflation, credit
    rate_series = load_fred_series("data/treasury.csv", value_col="DGS10")
    infl_series = load_fred_series("data/cpi.csv", value_col="CPIAUCSL")

    rate_label = "10Y Treasury (DGS10)"
    infl_label = "CPI (CPIAUCSL)"
    section_series = {}

    if rate_series is not None:
        section_series[rate_label] = rate_series.loc[start:end].ffill()
    else:
        rates_proxy = load_ticker_prices(["TLT"], start=start, end=end)
        if "TLT" in rates_proxy.columns:
            section_series["TLT (Rates proxy)"] = rates_proxy["TLT"]

    if infl_series is not None:
        section_series[infl_label] = infl_series.loc[start:end].ffill()
    else:
        infl_proxy = load_ticker_prices(["TIP"], start=start, end=end)
        if "TIP" in infl_proxy.columns:
            section_series["TIP (Inflation proxy)"] = infl_proxy["TIP"]

    credit_prices = load_ticker_prices(["LQD", "HYG"], start=start, end=end)
    if "LQD" in credit_prices.columns:
        section_series["LQD (IG Credit)"] = credit_prices["LQD"]
    if "HYG" in credit_prices.columns:
        section_series["HYG (HY Credit)"] = credit_prices["HYG"]

    rates_df = pd.DataFrame(section_series).ffill().dropna(how="all")
    if not rates_df.empty and freq == "Weekly":
        rates_df = rates_df.resample("W-FRI").last()
    rates_norm = normalize_to_100(rates_df) if not rates_df.empty else pd.DataFrame()
    # CPI can dwarf other series; plot CPI on a secondary axis for visibility.
    cpi_name = infl_label if infl_label in rates_norm.columns else None
    rates_fig = go.Figure()
    if not rates_norm.empty:
        for col in rates_norm.columns:
            if col == cpi_name:
                continue
            rates_fig.add_trace(go.Scatter(
                x=rates_norm.index,
                y=rates_norm[col],
                mode="lines",
                name=col,
                line={"width": 2},
            ))
    if cpi_name and cpi_name in rates_df.columns:
        rates_fig.add_trace(go.Scatter(
            x=rates_df.index,
            y=rates_df[cpi_name],
            mode="lines",
            name=f"{cpi_name} (right axis)",
            yaxis="y2",
        ))
    rates_fig.update_layout(
        title="Rates / Inflation / Credit (Rebased to 100; CPI on right axis)",
        legend_title_text="",
        height=420,
        autosize=False,
        yaxis=dict(title="Index (start=100)"),
        yaxis2=dict(title="CPI Level", overlaying="y", side="right"),
    )

    rates_rows = []
    for name, series in section_series.items():
        s = series.dropna()
        if s.empty:
            continue
        if freq == "Weekly":
            s = s.resample("W-FRI").last()
        row = {
            "series": name,
            "last": f"{s.iloc[-1]:.2f}",
            "change_1m": _format_pct(_period_change(s, 1)),
            "change_3m": _format_pct(_period_change(s, 3)),
            "change_ytd": _format_pct(_ytd_change(s)),
        }
        rates_rows.append(row)

    # Section C: heatmap
    ret_frames = {
        "Portfolio": port_ret,
    }
    if benchmark_prices is not None and not benchmark_prices.empty:
        ret_frames[benchmark] = bench_ret
    for col in proxy_prices.columns:
        ret_frames[col] = compute_returns(proxy_prices[col], freq)

    ret_df = pd.DataFrame(ret_frames).dropna(how="any")
    if ret_df.empty:
        heatmap_fig = _empty_fig("Not enough data for correlation heatmap.")
        heatmap_fig.update_layout(height=420, autosize=False)
    else:
        corr_matrix = ret_df.corr()
        heatmap_fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Correlation Heatmap",
        )
        heatmap_fig.update_layout(height=420, autosize=False)

    # Exposure summary table
    exposure_rows = []
    for col in proxy_prices.columns:
        proxy_ret = compute_returns(proxy_prices[col], freq)
        aligned = pd.concat([port_ret, proxy_ret], axis=1).dropna()
        if aligned.empty:
            continue
        beta, r2 = beta_and_r2(aligned.iloc[:, 0], aligned.iloc[:, 1])
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        proxy_prices_source = proxy_prices_chart[col].dropna() if col in proxy_prices_chart.columns else proxy_prices[col].dropna()
        ret_3m = _period_change(proxy_prices_source, 3)
        ann_vol = proxy_ret.std() * np.sqrt(ann_factor) if not proxy_ret.empty else np.nan
        exposure_rows.append({
            "proxy": col,
            "corr": f"{corr:.2f}" if pd.notna(corr) else "n/a",
            "beta": f"{beta:.2f}" if pd.notna(beta) else "n/a",
            "r2": f"{r2:.2f}" if pd.notna(r2) else "n/a",
            "ret_3m": _format_pct(ret_3m),
            "ann_vol": _format_pct(ann_vol),
        })

    exposure_rows = sorted(
        exposure_rows,
        key=lambda r: abs(float(r["beta"])) if r["beta"] not in ("n/a", "nan") else -1,
        reverse=True,
    )

    return (
        vol_fig,
        corr_fig,
        beta_fig,
        dd_fig,
        vol_metric,
        corr_metric,
        beta_metric,
        dd_metric,
        rates_fig,
        rates_rows,
        heatmap_fig,
        exposure_rows,
    )
