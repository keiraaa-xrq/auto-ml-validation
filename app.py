from dash.dependencies import Input, Output, State
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

from auto_ml_validation.app.index import app
from auto_ml_validation.app import header
from auto_ml_validation.app.callbacks import home_callbacks, results_callbacks
########################## Navbar ##########################
navbar = header.NavBar(app)


########################## Body ##########################
content = html.Div(id='page-content',
                   # children=[results_callbacks.results_layout],
                   children=[home_callbacks.home_layout],
                   )

########################## Callback ##########################
# Callback to show home page when the app is loaded


@app.callback(Output("page-content", "children"),
              [Input("url", "pathname")])
def display_page(pathname):
    print(pathname)
    if pathname == "/home" or pathname == "/":
        return home_callbacks.home_layout
    elif pathname == "/results":
        return results_callbacks.results_layout
    else:
        return html.P('Page not found')


########################## Layout ##########################
app.layout = html.Div([
    navbar,
    content,
    dcc.Location(id='url', refresh=False),
    # For data sharing between pages
    # Boolean to act as a trigger for result page callbacks
    dcc.Store(id='validator-input-trigger',
              data=False, storage_type='session'),
    # File name for convenient lookup
    dcc.Store(id='validator-input-file',
              data='', storage_type='session'),
    dcc.Store(id='validator-rep-model', data='', storage_type='session'),
    dcc.Store(id='validator-bm-model', data='', storage_type='session'),
])


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload=False)  # TODO
