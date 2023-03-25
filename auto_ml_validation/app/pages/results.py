import dash
from dash import html, dcc


results_layout = html.Div(children=[
    html.H1(children='This is our Results page'),

    html.Div(children='''
        This is our Result page content.
    '''),
    html.Div([
        # New Heading
        html.Div([
            html.H5('Probability Stability Index', style={'textAlign': 'left', 'fontWeight': 'bold'})
        ], style={'backgroundColor': '#ffaf77', 'width': '100%', 'top': 0, 'left': 0, 'margin': 0}),

        html.Div([
            html.H5('PSI Score', style={'textAlign': 'left', 'fontWeight': 'bold'})
        ], style={'backgroundColor': '#ffaf77', 'width': '100%', 'top': 0, 'left': 0, 'margin': 0}),
    ])

])
