# Navigation Bar Structure
# Maribank Logo and Application Title

from dash import html
import dash_bootstrap_components as dbc

def NavBar(app):
    navbar = dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=app.get_asset_url("images/MariBank.png"), height="200px", width = '250px')),
                        dbc.Col(html.H2(app.title, style={"color": "#ec6b12", "font-weight":"bold"}, className="navbar-brand")),
                    ],
                    align="center"
                )
            ]
        ),
        color="white",
        dark=False,
        sticky="top",
        style={
            "display": "flex",
            "justify-content": "center",
            "align-items": "center",
            "height": "10vh",
        }
    )
    
    return navbar
