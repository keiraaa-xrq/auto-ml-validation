import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

def generate():
    return html.Div([
        html.H1(children=project_name),
        html.Div([
            html.H6(f'Model algorithm: {algo}'),
            html.H6(f'Testing sample size: {sample_size}'),
            html.Div([
                html.H6('Classification threshold: '),
                html.Div([
                    html.Div([dcc.Input(id='threshold', type='range', value=0.5, min=0, max=1)]),
                    # dcc.Store(id='threshold-store'),
                    # html.H6(id='threshold-store')
                ])
            ])
        ])    ]),

    # html.Div(dbc.Tabs(id="tabs", children=[
    #         dbc.Tab(label="Metrics", tab_id="tab-1"),
    #         dbc.Tab(label="FEAT", tab_id="tab-2")
    #     ])),

    dcc.Graph(id='dist-curve', figure={}),
    dcc.Graph(id='roc-curve', figure={}),
    dcc.Graph(id='pr-curve', figure={}),
    html.Div(id='metrics')