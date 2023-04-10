import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from auto_ml_validation.validation_package.evaluation import statistical_metrics_evaluator
from auto_ml_validation.validation_package.evaluation import transparency_metrics_evaluator
from auto_ml_validation.validation_package.evaluation import performance_metrics_evaluator
import pickle
from sklearn.linear_model import LogisticRegression
from auto_ml_validation.app.index import app
import matplotlib.pyplot as plt


def sticky_headers():
    bm_header = dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.H2(
                                'Result for Benchmark Model',
                                style={
                                    'font-weight': 'bold',
                                    'font-size': '30px',
                                    'margin-top': '20px',
                                    'margin-bottom': '10px',
                                    'text-transform': 'uppercase',
                                    'letter-spacing': '1px',
                                    'color': '#333333',
                                    'text-align': 'center'
                                },
                            ),
                        ),
                    ],
                    align='center',
                )
            ]
        ),
        color='white',
        dark=False,
        sticky='top',
        style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center',
            'height': '10vh',
        },
    )

    re_header = dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.H2(
                                'Result for Replicated Model',
                                style={
                                    'font-weight': 'bold',
                                    'font-size': '30px',
                                    'margin-top': '20px',
                                    'margin-bottom': '10px',
                                    'text-transform': 'uppercase',
                                    'letter-spacing': '1px',
                                    'color': '#333333',
                                },
                            ),
                        ),
                    ],
                    align='center',
                )
            ]
        ),
        color='white',
        dark=False,
        sticky='top',
        style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center',
            'height': '10vh',
        },
    )
    return re_header, bm_header


def bm_performance_metric_layout():
    return html.Div(style={
        'border': '2px solid black',
        'padding': '20px',
        'background-color': '#FFFBFA',
        'width': '100%',
        'margin': 'auto',
        'text-align': 'center'
    }, children=[
        html.H2('Model Related Measures', style={'font-weight': 'bold'}),
        html.H6('Probability distribution from model prediction output.'),
        dcc.Loading(id="loading-dist-curve", type="circle",
                    style={'display': 'flex', 'justify-content': 'center'}, children=[
                        dcc.Graph(id='dist-curve', figure={})]),
        html.H6('Shows the performance of the model at all classification thresholds. Higher AUC indicates a better model.'),
        dcc.Loading(id="loading-roc-curve", type="circle",
                    style={'display': 'flex', 'justify-content': 'center'}, children=[
                        dcc.Graph(id='roc-curve', figure={}, style={'margin': 'auto'})]),
        html.H6('Shows the tradeoff between precision and recall for different thresholds ' +
                'in binary classification problems. A high area under the curve represents both ' +
                'high recall and high precision, where high precision relates to a low false ' +
                'positive rate, and high recall relates to a low false negative rate. '),
        dcc.Loading(id="loading-pr-curve", type="circle",
                    style={'display': 'flex', 'justify-content': 'center'}, children=[
                        dcc.Graph(id='pr-curve', figure={}, style={'margin': 'auto'})]),
        html.H6(id='threshold-text', children='Adjust the threshold here: 0.5',
                style={'font-weight': 'bold'}),
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
        html.H2('Model Related Measures', style={'font-weight': 'bold'}),
        html.H6('Probability distribution from model prediction output.'),
        dcc.Loading(id="loading-dist-curve-re", type="circle",
                    style={'display': 'flex', 'justify-content': 'center'}, children=[
                        dcc.Graph(id='dist-curve-re', figure={})]),
        html.H6('Shows the performance of the model at all classification thresholds. Higher AUC indicates a better model.'),
        dcc.Loading(id="loading-roc-curve-re", type="circle",
                    style={'display': 'flex', 'justify-content': 'center'}, children=[
                        dcc.Graph(id='roc-curve-re', figure={}, style={'margin': 'auto'})]),
        html.H6('Shows the tradeoff between precision and recall for different thresholds ' +
                'in binary classification problems. A high area under the curve represents both ' +
                'high recall and high precision, where high precision relates to a low false ' +
                'positive rate, and high recall relates to a low false negative rate. '),
        dcc.Loading(id="loading-pr-curve-re", type="circle",
                    style={'display': 'flex', 'justify-content': 'center'}, children=[
                        dcc.Graph(id='pr-curve-re', figure={}, style={"margin": "auto", "display": "flex", "justify-content": "center"})]),
        html.H6(id='threshold-text-re', children='Adjust the threshold here: 0.5',
                style={'font-weight': 'bold'}),
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
        html.H3('Probability Stability Index Table',
                style={'font-weight': 'bold'}),
        html.H5(id='psi-score',
                children='',
                style={'font-weight': 'bold'}),
        html.H6('A metric to measure how much a variable has shifted in distribution between two samples over time. '),
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
                             style_cell={'minWidth': 0, 'maxWidth': 50, 'whiteSpace': 'normal'}),
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
        html.H3('Probability Stability Index Table',
                style={'font-weight': 'bold'}),
        html.H5(id='psi-score-re',
                children='',
                style={'font-weight': 'bold'}),
        html.H6('A metric to measure how much a variable has shifted in distribution between two samples over time. '),
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
                             style_cell={'minWidth': 0, 'maxWidth': 50, 'whiteSpace': 'normal'}),
        html.Br(),
        html.Div(id='ks-tests-re', children=None, style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center',
            'flex-direction': 'column',
        }),
        html.Br(),
    ])


def bm_gini_layout() -> html.Div:
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
            html.H2('Feature Related Measures', style={'font-weight': 'bold'}),
            html.H3('GINI Index',  style={
                'textAlign': 'center',
                'fontWeight': 'bold',
                'margin-bottom': '20px'  # add margin below title
            }),
            html.H6('A measurement used to build Decision Trees to determine how the features of a dataset' +
                    ' should split nodes to form the tree. It calculates the probability of a certain randomly ' +
                    'selected feature that is classified incorrectly. An attribute with the smallest Gini Impurity' +
                    ' should be selected for splitting the node.'),
            html.Label('Select Features to Display:',
                       style={'font-weight': 'bold'}),
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


def re_gini_layout() -> html.Div:
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
            html.H2('Feature Related Measures', style={'font-weight': 'bold'}),
            html.H3('GINI Index',  style={
                'textAlign': 'center',
                'fontWeight': 'bold',
                'margin-bottom': '20px'  # add margin below title
            }),
            html.H6('A measurement used to build Decision Trees to determine how the features of a dataset' +
                    ' should split nodes to form the tree. It calculates the probability of a certain randomly ' +
                    'selected feature that is classified incorrectly. An attribute with the smallest Gini Impurity' +
                    ' should be selected for splitting the node.'),
            html.Label('Select Features to Display:',
                       style={'font-weight': 'bold'}),
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


def bm_csi_table_layout() -> html.Div:
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
            html.H6('CSI compares the distribution of an independent variable in the training data set ' +
                    'to a testing data set. It detects shifts in the distributions of input variables that ' +
                    'are submitted for scoring over time.'),
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
                html.H4('Characteristic Stability Index Table', style={
                        'text-align': 'center', 'font-weight': 'bold'})
            ], style={'margin': '0'}),
            html.Div(id="feature-related-viz", children=[]),
            html.Br(),
            html.Br()
        ]
    )


def re_csi_table_layout() -> html.Div:
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
                'margin-bottom': '20px'
            }),
            html.H6('CSI compares the distribution of an independent variable in the training data set ' +
                    'to a testing data set. It detects shifts in the distributions of input variables that ' +
                    'are submitted for scoring over time.'),
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
                html.H4('Characteristic Stability Index Table', style={
                        'text-align': 'center', 'font-weight': 'bold'})
            ], style={'margin': '0'}),
            html.Div(id="feature-related-viz-re", children=[]),
            html.Br(),
            html.Br()
        ]
    )


def bm_trans_layout() -> html.Div:
    img_style = {'width': '600px', 'height': '400px', 'object-fit': 'contain'}
    return html.Div(style={
        'background-color': '#F4F8FD',
        'border': '2px solid black',
        'padding': '20px',
        'width': '100%',
        'margin': 'auto',
        'text-align': 'center'
    }, children=[
        html.H2('Transparency Metrics', style={
                'text-align': 'center', 'font-weight': 'bold'}),
        html.H6('''The SHAP plot shows the average impact of each 
        feature on the model's output across all instances in the dataset. The features are sorted 
        in descending order of importance, and the magnitude and direction of the impact are shown. \n 
        '''),
        html.Br(),
        html.H4('Global Lime Plot', style={'fontWeight': 'bold'}),
        dcc.Loading(
            id="loading-global-lime",
            type="circle",
            style={'display': 'flex', 'justify-content': 'center'},
            children=[html.Img(id='global-lime', src='', style=img_style)]
        ),
        html.H4('Global Shap Plot', style={'fontWeight': 'bold'}),
        dcc.Loading(
            id="loading-global-shap",
            type="circle",
            style={'display': 'flex', 'justify-content': 'center'},
            children=[html.Img(id='global-shap', src='', style=img_style)]
        ),
    ])


def re_trans_layout() -> html.Div:
    img_style = {'width': '600px', 'height': '400px', 'object-fit': 'contain'}
    return html.Div(style={
        'background-color': '#F4F8FD',
        'border': '2px solid black',
        'padding': '20px',
        'width': '100%',
        'margin': 'auto',
        'text-align': 'center'
    }, children=[
        html.H2('Transparency Metrics', style={
                'text-align': 'center', 'font-weight': 'bold'}),
        html.H6('The LIME chart shows the coefficients assigned ' +
                'to each feature by the interpretable model. The coefficients indicate the direction ' +
                'and magnitude of the impact of each feature on the model''s output. ' +
                'Positive coefficients indicate that the feature has a positive impact on the Class1, ' +
                'while negative coefficients indicate a positive impact on Class 0.                   '),
        html.Br(),
        html.H4('Global Lime Plot', style={'fontWeight': 'bold'}),
        dcc.Loading(
            id="loading-global-lime-re",
            type="circle",
            style={'display': 'flex', 'justify-content': 'center'},
            children=[html.Img(id='global-lime-re', src='', style=img_style)]
        ),
        html.H4('Global Shap Plot', style={'fontWeight': 'bold'}),
        dcc.Loading(
            id="loading-global-shap-re",
            type="circle",
            style={'display': 'flex', 'justify-content': 'center'},
            children=[html.Img(id='global-shap-re', src='', style=img_style)]
        ),
    ])


def download_report_layout():
    return html.Div([
        dbc.Button(
            "Download Report",
            id="download-report",
            color="primary",
            className="mx-auto d-block",
            style={
                "marginTop": "20px",
                "textAlign": "center",
                "border": "none",
                "color": "white",
                "padding": "10px 20px",
                "text-align": "center",
                "text-decoration": "none",
                "display": "inline-block",
                "font-size": "16px",
                "border-radius": "4px",
                "margin-right": "60px"
            }
        ),
        html.Div(
            [
                dbc.Spinner(
                    html.Div(
                        id="report-message", style={"margin-top": "15px", "text-align": "center"}),
                    size="lg",
                    color="primary",
                    type="grow"
                )
            ],
            style={"margin-right": "30px", "margin-top": "10px"}
        ),
    ])
