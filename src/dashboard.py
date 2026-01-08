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
for path in NAV_ORDER:
    page = page_by_path.get(path)
    if not page:
        continue
    nav_links.append(
        dbc.NavItem(
            dbc.NavLink(page["name"], href=page["path"], active="exact")
        )
    )

app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="Schwab Portfolio Dashboard",
        color="primary",
        dark=True,
        children=nav_links
    ),
    html.Div(dash.page_container, className="page-container"),
], fluid=True, className="app-shell")

# Run app
if __name__ == "__main__":
    app.run(debug=True)
