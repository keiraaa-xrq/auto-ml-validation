import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

############## local variable ##################
project_name = 'Credit Risk'
algo = "Logistic Regression"
sample_size = 1000


def result_control():
    return html.Div([
        html.H1(children=project_name),
        html.Div([
            html.H6(f'Model algorithm: {algo}'),
            html.H6(f'Testing sample size: {sample_size}'),
            html.Div([
                html.H6('Classification threshold: '),
                html.Div([
                    html.Div(
                        [dcc.Input(id='threshold', type='range', value=0.5, min=0, max=1)]),
                ])
            ])
        ]),

        html.Div(dbc.Tabs(id="tabs", children=[
            dbc.Tab(label="Metrics", tab_id="tab-1"),
            dbc.Tab(label="FEAT", tab_id="tab-2")
        ])),
    ])


def model_metrics():
    return html.Div([
        dcc.Graph(id='dist-curve', figure={}),
        
        html.Div(id='confusion'),
        html.Div(id='metrics'),

        dcc.Graph(id='roc-curve', figure={}),
        dcc.Graph(id='pr-curve', figure={}),
        dcc.Graph(id='lift-curve', figure={})
    ])
