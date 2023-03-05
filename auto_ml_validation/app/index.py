from dash import Dash
import dash_bootstrap_components as dbc

app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.UNITED])
app.title = 'Model Validation Tool'