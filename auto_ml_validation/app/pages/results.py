import dash
from dash import html, dcc, dash_table
import pandas as pd
import numpy as np
from auto_ml_validation.validation_package.evaluation import statistical_metrics_evaluator
from auto_ml_validation.validation_package.evaluation import transparency_metrics_evaluator
from auto_ml_validation.validation_package.evaluation import performance_metrics_evaluator
import pickle
from sklearn.linear_model import LogisticRegression
from auto_ml_validation.app.index import app
import matplotlib.pyplot as plt

def bm_performance_metric_layout():
    return html.Div(style={
        'border': '2px solid black',
        'padding': '20px',
        'background-color': '#FFFBFA',
        'width': '100%',
        'margin': 'auto',
        'text-align': 'center'
    }, children=[ 
        html.H2('Global Performance Measures', style={'font-weight': 'bold'}),
        html.Div(style={'display': 'flex', 'justify-content': 'space-between'}, children=[
            dcc.Graph(id='dist-curve', figure={})]),
        html.Div(style={'display': 'flex', 'justify-content': 'space-between'}, children=[
            dcc.Graph(id='roc-curve', figure={})]),
        html.Div(style={'display': 'flex', 'justify-content': 'space-between'}, children=[
            dcc.Graph(id='pr-curve', figure={})]),
        html.H6(id='threshold-text', children='Adjust the threshold here: 0.5', style={'font-weight': 'bold'}),
        dcc.Input(id='threshold', type='range', value=0.5, min=0, max=1),
        html.Div(id='metrics', style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center',
            'flex-direction': 'column',
        })       
        ])

def re_performance_metric_layout():
    return html.Div(style={
        'border': '2px solid black',
        'padding': '20px',
        'background-color': '#FFFBFA',
        'width': '100%',
        'margin': 'auto',
        'text-align': 'center'
    }, children=[ 
        html.H2('Global Performance Measures', style={'font-weight': 'bold'}),
        html.Div(style={'display': 'flex', 'justify-content': 'space-between'}, children=[
            dcc.Graph(id='dist-curve-re', figure={})]),
        html.Div(style={'display': 'flex', 'justify-content': 'space-between'}, children=[
            dcc.Graph(id='roc-curve-re', figure={})]),
        html.Div(style={'display': 'flex', 'justify-content': 'space-between'}, children=[
            dcc.Graph(id='pr-curve-re', figure={})]),
        html.H6(id='threshold-text-re',children='0.5', style={'font-weight': 'bold'}),
        dcc.Input(id='threshold-re', type='range', value=0.5, min=0, max=1),
        html.Div(id='metrics-re', style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center',
            'flex-direction': 'column',
        })       
        ])

def bm_statistical_model_metrics_layout() -> html.Div:
    return html.Div(style={
        'border': '2px solid black',
        'background-color': '#FFFBFA',
        'width': '100%',
        'margin': 'auto',
        'text-align': 'center',
        'padding': '20px'
    }, children=[
        html.H3('Probability Stability Index Table', style={'font-weight': 'bold'}),
        html.H5(id='psi-score',
                children='',
                style={'font-weight': 'bold'}),
        html.Br(),
        html.Label('Please choose the number of bins to be sliced: '),
        dcc.Input(id='psi-num-of-bins',
                  type='number',
                  value=10,
                  min=2,
                  max=50,
                  step=1,
                  pattern=r'\d*'),
        html.Br(),
        dash_table.DataTable(id='psi-table',
                              data=None,
                              columns=None,
                              sort_action='native',
                              style_cell={'minWidth': 0, 'maxWidth': 50, 'whiteSpace': 'normal' }),
        html.Br(),
        html.Div(id='ks-tests', children=None, style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center',
            'flex-direction': 'column',
        }),
        html.Br(),
    ])


def re_statistical_model_metrics_layout() -> html.Div:
    return html.Div(style={
        'border': '2px solid black',
        'background-color': '#FFFBFA',
        'width': '100%',
        'margin': 'auto',
        'text-align': 'center',
        'padding': '20px'
    }, children=[
        html.H3('Probability Stability Index Table', style={'font-weight': 'bold'}),
        html.H5(id='psi-score-re',
                children='',
                style={'font-weight': 'bold'}),
        html.Br(),
        html.Label('Please choose the number of bins to be sliced: '),
        dcc.Input(id='psi-num-of-bins-re',
                  type='number',
                  value=10,
                  min=2,
                  max=50,
                  step=1,
                  pattern=r'\d*'),
        html.Br(),
        dash_table.DataTable(id='psi-table-re',
                              data=None,
                              columns=None,
                              sort_action='native',
                              style_cell={'minWidth': 0, 'maxWidth': 50, 'whiteSpace': 'normal' }),
        html.Br(),
        html.Div(id='ks-tests-re', children=None, style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center',
            'flex-direction': 'column',
        }),
        html.Br(),
    ])

def bm_gini_layout()-> html.Div:
    return html.Div(
        style={
            'border': '2px solid black',
            'padding': '20px',
            'width': '100%',
            'margin': 'auto',
            'background-color': '#EFEFEF',
            'text-align': 'center'
        },
        children=[
            html.Br(),
            html.H3('GINI Index',  style={
                'textAlign': 'center', 
                'fontWeight': 'bold',
                'margin-bottom': '20px'  # add margin below title
            }),
            html.Label('Select Features to Display:', style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id="gini-feature-multi-dynamic-dropdown", 
                multi=True,
                options={},
                value=[],
                style={'width': '50%', 'margin': 'auto'}
            ), 
            html.Br(),
            html.Br(),
            dcc.Loading(
                id="loading-gini",
                type="circle",
                children=html.Div([
                    # Output of long-running task
                    html.Div(id="gini-viz", style={'fontWeight': 'bold'}),
                ]),
            ),
            html.Br()       
        ]
    )

def re_gini_layout()-> html.Div:
    return html.Div(
        style={
            'border': '2px solid black',
            'padding': '20px',
            'width': '100%',
            'margin': 'auto',
            'background-color': '#EFEFEF',
            'text-align': 'center'
        },
        children=[
            html.Br(),
            html.H3('GINI Index',  style={
                'textAlign': 'center', 
                'fontWeight': 'bold',
                'margin-bottom': '20px'  # add margin below title
            }),
            html.Label('Select Features to Display:', style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id="gini-feature-multi-dynamic-dropdown-re", 
                multi=True,
                options={},
                value=[],
                style={'width': '50%', 'margin': 'auto'}
            ), 
            html.Br(),
            html.Br(),
            dcc.Loading(
                id="loading-gini-re",
                type="circle",
                children=html.Div([
                    # Output of long-running task
                    html.Div(id="gini-viz-re", style={'fontWeight': 'bold'}),
                ]),
            ),
            html.Br()       
        ]
    )


def bm_csi_table_layout()-> html.Div:
    return html.Div(
        style={
            'background-color': '#EFEFEF',
            'border': '2px solid black',
            'padding': '20px',
            'width': '100%',
            'margin': 'auto',
            'text-align': 'center'
        }, 
        children=[
            html.H3('CSI Metrics', style={
                'textAlign': 'center', 
                'fontWeight': 'bold',
                'margin-bottom': '20px'  # add margin below title
            }),
            html.Br(),
            html.Label('Features to Display:', style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id="csi-feature-multi-dynamic-dropdown", 
                multi=True,
                options={},
                value=[],
                style={'width': '50%', 'margin': 'auto'}
            ),
            html.Br(),
            html.Label('Number of Bins:', style={'font-weight': 'bold'}),
            dcc.Input(
                id="csi-num-of-bins",
                type='number',
                value=10,
                min=2,
                max=50,
                step=1,
                pattern=r'\d*',
                style={'width': '20%', 'margin': 'auto'}
            ),
            html.Br(),
            html.Br(),
            html.Div([
                html.H4('Characteristic Stability Index Table', style={'text-align': 'center', 'font-weight': 'bold'})
            ], style={'margin': '0'}),
            html.Div(id="feature-related-viz", children = []),
            html.Br(),
            html.Br()        
        ]
    )

def re_csi_table_layout()-> html.Div:
    return html.Div(
        style={
            'background-color': '#EFEFEF',
            'border': '2px solid black',
            'padding': '20px',
            'width': '100%',
            'margin': 'auto',
            'text-align': 'center'
        }, 
        children=[
            html.H3('CSI Metrics', style={
                'textAlign': 'center', 
                'fontWeight': 'bold',
                'margin-bottom': '20px'  # add margin below title
            }),
            html.Br(),
            html.Label('Features to Display:', style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id="csi-feature-multi-dynamic-dropdown-re", 
                multi=True,
                options={},
                value=[],
                style={'width': '50%', 'margin': 'auto'}
            ),
            html.Br(),
            html.Label('Number of Bins:', style={'font-weight': 'bold'}),
            dcc.Input(
                id="csi-num-of-bins-re",
                type='number',
                value=10,
                min=2,
                max=50,
                step=1,
                pattern=r'\d*',
                style={'width': '20%', 'margin': 'auto'}
            ),
            html.Br(),
            html.Br(),
            html.Div([
                html.H4('Characteristic Stability Index Table', style={'text-align': 'center', 'font-weight': 'bold'})
            ], style={'margin': '0'}),
            html.Div(id="feature-related-viz-re", children = []),
            html.Br(),
            html.Br()        
        ]
    )
    

def bm_trans_layout()->html.Div:
    img_style = {'width': '600px', 'height': '400px', 'object-fit': 'contain'}
    return html.Div(style={
        'border': '1px solid black',
        'padding': '10px',
        'margin': '10px auto',
        'max-width': '800px',
        'background-color': '#FFD580',
        'text-align': 'center'
    }, children=[html.H2('Transparency Metrics', style={'text-align': 'center'}),
                 html.H4('Local Lime Plot', style={'fontWeight': 'bold'}),
                 html.Img(id='local-lime', src=app.get_asset_url("images/local_lime_bm.png"), style=img_style),
                 html.H4('Global Lime Plot', style={'fontWeight': 'bold'}),
                 html.Img(id='global-lime', src=app.get_asset_url("images/global_lime_bm.png"), style=img_style),
                 html.H4('Local Shap Plot', style={'fontWeight': 'bold'}),
                 html.Img(id='local-shap', src=app.get_asset_url("images/local_shap_bm.png"), style=img_style),
                 html.H4('Global Shap Plot', style={'fontWeight': 'bold'}),
                 html.Img(id='global-shap', src=app.get_asset_url("images/global_shap_bm.png"),style=img_style),
                 ])

def re_trans_layout()->html.Div:
    img_style = {'width': '600px', 'height': '400px', 'object-fit': 'contain'}
    return html.Div(style={
        'border': '1px solid black',
        'padding': '10px',
        'margin': '10px auto',
        'max-width': '800px',
        'background-color': '#FFD580',
        'text-align': 'center'
    }, children=[html.H2('Transparency Metrics', style={'text-align': 'center'}),
                 html.H4('Local Lime Plot', style={'fontWeight': 'bold'}),
                 html.Img(id='local-lime-re', src=app.get_asset_url("images/local_lime_re.png"), style=img_style),
                 html.H4('Global Lime Plot', style={'fontWeight': 'bold'}),
                 html.Img(id='global-lime-re', src=app.get_asset_url("images/global_lime_re.png"), style=img_style),
                 html.H4('Local Shap Plot', style={'fontWeight': 'bold'}),
                 html.Img(id='local-shap-re', src=app.get_asset_url("images/local_shap_re.png"), style=img_style),
                 html.H4('Global Shap Plot', style={'fontWeight': 'bold'}),
                 html.Img(id='global-shap-re', src=app.get_asset_url("images/global_shap_re.png"),style=img_style),
                 ])
