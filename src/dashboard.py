import dash
from dash import Dash, html
import dash_bootstrap_components as dbc

# Initialize app
app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Schwab Portfolio Dashboard"

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
        brand_style={"fontSize": "1.5rem", "fontWeight": "600"},
        children=dbc.Nav(
            nav_links,
            navbar=True,
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(5, minmax(150px, 1fr))",
                "rowGap": "6px",
                "columnGap": "18px",
                "marginTop": "0px",
                "width": "100%",
            },
        ),
    ),
    html.Div(dash.page_container, className="page-container"),
], fluid=True, className="app-shell")

# Run app
if __name__ == "__main__":
    app.run(debug=True)
