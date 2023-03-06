import base64
import io
from typing import *
import json
import time
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from auto_ml_validation.validation_package.train_pipeline import train_pipeline


external_stylesheets = [
    {
        'href': 'https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-rbsA2VBKQhggwzxH7pPCaAqO46MgnOM80zW1RWuH61DGLwZJEdK2Kadq2F9CUG65',
        'rel': 'stylesheet',
        'crossorigin': 'anonymous'
    }
]

algos = {
    'logistic': 'Logistic Regression',
    'svm': 'Support Vector Machines',
    'knn': 'K Nearest Neighbors',
    'dt': 'Decision Tree',
    'rf': 'Random Forest',
    'xgboost': 'XGBoost'
}

model = None


def display_confusion_matrix(df):
    table = html.Table([
        html.Thead(
            html.Tr([html.Th(), html.Th(df.columns[0]), html.Th(df.columns[1])])
        ),
        html.Tbody([
            html.Tr([html.Th(df.index[0]), html.Td(
                df.iloc[0, 0]), html.Td(df.iloc[0, 1])]),
            html.Tr([html.Th(df.index[1]), html.Td(
                df.iloc[1, 0]), html.Td(df.iloc[1, 1])])
        ])
    ], className='table')
    return table


def display_dict(dictionary):
    return html.Ul([
        html.Li(f'{k}: {v}', className='list-group-item') for k, v in dictionary.items()
    ], className='list-group')


app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1('Model Training Prototype'),

    # form for setting up project
    html.H2('Project Configuration'),
    html.Div([
        html.H3('Datasets'),
        # upload training data
        html.Div([
            html.Label('Upload training dataset (*.csv):',
                       className='form-label'),
            html.P(id='train-name', className='fw-bold'),
            dcc.Upload(
                html.A('Select File'),
                id='upload-train',
                className='btn btn-outline-primary'
            ),
            dcc.Store(id='train-data')
        ], className='m-3'),
        # upload testing data
        html.Div([
            html.Label('Upload testing dataset (*.csv):',
                       className='form-label'),
            html.P(id='test-name', className='fw-bold'),
            dcc.Upload(
                html.A('Select File'),
                id='upload-test',
                className='btn btn-outline-primary'
            ),
            dcc.Store(id='test-data')
        ], className='m-3'),
        # select target variable
        html.Div([
            html.Label('Select target variable:', className='form-label'),
            dcc.Dropdown(id='target')
        ], className='m-3'),

        html.H3('Algorithm'),
        # select algo name
        html.Div([
            html.Label('Select algorithm:', className='form-label'),
            dcc.Dropdown(
                options=algos,
                id='algo'
            )
        ], className='m-3'),
        # upload hyperparameter settings
        html.Div([
            html.Label('Upload hyperparameter values (*.json):',
                       className='form-label'),
            html.P(id='params-name', className='fw-bold'),
            dcc.Upload(
                html.A('Select File'),
                id='upload-params',
                className='btn btn-outline-primary'
            ),
            dcc.Store(id='params')
        ], className='m-3'),
        # submit
        html.Div([
            html.Button('Submit', n_clicks=0, id='submit', type='submit',
                        className='btn btn-primary btn-lg')
        ], className='container m-3 text-center')
    ], className='container border rounded'),

    # show training outcome
    html.Br(),
    html.Div([
        html.Div(id='start-train'),
        dcc.Loading(
            html.P(id='complete-train'),
            type="circle",
        ),
    ])
], style={'padding': 40, 'fontFamily': 'sans-serif'}
)


def parse_contents(contents):
    '''
    Decode uploaded contents.
    '''
    # print(contents)
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    # print(decoded)
    return decoded


def update_data(contents, filename):
    if contents is not None:
        decoded = parse_contents(contents)
        # TODO: handle parsing error
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        df_json = df.to_json(date_format='iso', orient='split')
        return filename, df_json
    else:
        return 'No file selected.', {}


@app.callback(
    Output('train-name', 'children'),
    Output('train-data', 'data'),
    Input('upload-train', 'contents'),
    State('upload-train', 'filename'),
)
def update_train(contents, filename):
    return update_data(contents, filename)


@app.callback(
    Output('test-name', 'children'),
    Output('test-data', 'data'),
    Input('upload-test', 'contents'),
    State('upload-test', 'filename'),
)
def update_test(contents, filename):
    return update_data(contents, filename)


@app.callback(
    Output('target', 'options'),
    Input('train-data', 'data')
)
def select_target_dropdown(jsonified_data):
    if jsonified_data:
        df = pd.read_json(jsonified_data, orient='split')
        cols = df.columns.to_list()
        return cols
    else:
        return []


@app.callback(
    Output('params-name', 'children'),
    Output('params', 'data'),
    Input('upload-params', 'contents'),
    State('upload-params', 'filename')
)
def update_params(contents, filename):
    # TODO: handle parsing error
    if contents is not None:
        decoded = parse_contents(contents)
        params = json.loads(decoded)
        # print(type(params), params)
        return filename, params
    else:
        return 'No file selected.', {}


@app.callback(
    Output('start-train', 'children'),
    Input('submit', 'n_clicks')
)
def start_train(n):
    if n > 0:
        div = html.Div([
            html.H2('Model Training'),
            html.P('Performing training...')
        ])
        return div


@app.callback(
    Output('complete-train', 'children'),
    Input('submit', 'n_clicks'),
    State('train-data', 'data'),
    State('test-data', 'data'),
    State('target', 'value'),
    State('algo', 'value'),
    State('params', 'data'),
    prevent_initial_call=True
)
def train(n, train_data, test_data, target, algo, params):
    if n > 0:
        # convert jsonified data to df
        train_df = pd.read_json(train_data, orient='split')
        test_df = pd.read_json(test_data, orient='split')
        start = time.time()
        time.sleep(5)
        train_clf, train_metrics, test_clf, test_metrics = train_pipeline(
            train_df, test_df, target, algo, params
        )
        end = time.time()
        dur = (end - start)/60
        div = html.Div([
            html.P(f'Training completed! Time taken: {dur:.2f} mins.'),
            html.P('Performance on the training data:'),
            display_confusion_matrix(train_clf),
            display_dict(train_metrics),
            html.Br(),
            html.P('Performance on the testing data:'),
            display_confusion_matrix(test_clf),
            display_dict(test_metrics),
        ])
        return div


if __name__ == '__main__':
    app.run_server(debug=True)
