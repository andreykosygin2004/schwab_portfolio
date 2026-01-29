from pathlib import Path

import dash
from dash import Dash, html
import dash_bootstrap_components as dbc

# Initialize app
app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Schwab Portfolio Dashboard"
DATA_DIR = Path("data")
HAS_LOCAL_PORTFOLIO = (DATA_DIR / "schwab_transactions.csv").exists()

NAV_ORDER = [
    "/",
    "/benchmarks",
    "/risk",
    "/holdings-intel",
    "/attribution",
    "/factors",
    "/macro",
    "/regimes",
    "/regime-forecast",
    "/factor-rotation",
]

page_by_path = {page["path"]: page for page in dash.page_registry.values()}
nav_links = []
NAV_LINK_STYLE = {
    "textAlign": "center",
    "whiteSpace": "nowrap",
    "lineHeight": "1.1",
    "padding": "4px 8px",
    "width": "100%",
}
for path in NAV_ORDER:
    page = page_by_path.get(path)
    if not page:
        continue
    nav_links.append(
        dbc.NavItem(
            dbc.NavLink(page["name"], href=page["path"], active="exact", style=NAV_LINK_STYLE)
        )
    )

app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="Schwab Portfolio Dashboard",
        color="primary",
        dark=True,
        brand_style={"fontSize": "2rem", "fontWeight": "600", "marginLeft": "0"},
        style={"paddingLeft": "0", "paddingRight": "0"},
        fluid=True,
        children=dbc.Nav(
            nav_links,
            navbar=True,
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(5, minmax(150px, 1fr))",
                "rowGap": "6px",
                "columnGap": "12px",
                "marginTop": "0px",
                "marginLeft": "-6px",
                "width": "100%",
            },
        ),
    ),
    html.Div(
        [
            html.Label("Portfolio", style={"fontWeight": "600", "marginRight": "8px"}),
            dbc.Select(
                id="portfolio-selector",
                options=(
                    [
                        {"label": "My Portfolio", "value": "schwab"},
                        {"label": "Algory Portfolio", "value": "algory"},
                    ]
                    if HAS_LOCAL_PORTFOLIO
                    else [{"label": "Algory Portfolio", "value": "algory"}]
                ),
                value="schwab" if HAS_LOCAL_PORTFOLIO else "algory",
                style={"maxWidth": "240px"},
            ),
        ],
        style={"display": "flex", "alignItems": "center", "gap": "8px", "marginTop": "20px"},
    ),
    html.Div(dash.page_container, className="page-container"),
], fluid=True, className="app-shell")

# Run app
if __name__ == "__main__":
    app.run(debug=True)
