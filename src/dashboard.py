import dash
from dash import Dash
import dash_bootstrap_components as dbc

# Initialize app
app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Schwab Portfolio Dashboard"

nav_links = [
    dbc.NavItem(
        dbc.NavLink(page["name"], href=page["path"])
    )
    for page in dash.page_registry.values()
]

app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="Schwab Portfolio Dashboard",
        color="primary",
        dark=True,
        children=nav_links
    ),
    dash.page_container,
], fluid=True)

# Run app
if __name__ == "__main__":
    app.run(debug=True)