from dash.dependencies import Input, Output, State
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

from auto_ml_validation.app.index import app
from auto_ml_validation.app import header
from auto_ml_validation.app.callbacks import home_callbacks
from auto_ml_validation.app.pages.results import results_layout

########################## Navbar ##########################
navbar = header.NavBar(app)


########################## Body ##########################


content = html.Div(id='page-content', children=[])

########################## Callback ##########################
# Callback to show home page when the app is loaded


@app.callback(Output("page-content", "children"),
              [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/home":
        return home_callbacks.home_layout
    elif pathname == "/results":
        return dbc.Tabs(id="tabs", children=[
            dbc.Tab(label="Tab 1", tab_id="tab-1"),
            dbc.Tab(label="Tab 2", tab_id="tab-2")
        ])
    else:
        return html.P('Page not found')


########################## Layout ##########################
app.layout = html.Div([dcc.Location(id='url', refresh=False, pathname='/home'),
                       navbar,
                       content
                       ])


if __name__ == '__main__':
    app.run_server(debug=False)
