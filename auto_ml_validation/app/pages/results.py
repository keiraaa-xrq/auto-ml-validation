import dash
from dash import html, dcc


results_p_layout = html.Div(children=[
    html.H1(id ='output-div', children='This is our Results page'),

    html.Div(children='''
        This is our Result page content.
    '''),

])
