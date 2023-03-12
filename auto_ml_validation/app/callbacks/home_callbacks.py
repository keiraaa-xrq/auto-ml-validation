# Landing Page Callbacks
from auto_ml_validation.app.index import app
from auto_ml_validation.app.pages.home import *

import pandas as pd
import io
import base64
import json
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
##################### Variables Config #####################
# Map algorithms to their hyperparameters


# Body Content
form, submit_button = project_field()

# Layout
home_layout = html.Div([
    form,
    submit_button,   
    dcc.Store(id='store-project', data={}, storage_type='memory'), # Dictionary of Project Config {project name: value, algorithm: value}
    dcc.Store(id='store-rep-data', data = {}), # Dictionary of Replicating Model Data {Hyperparamaters: Value, Train Dataset: Value, Test Dataset: Value, Other Dataset: Value, Target: Value, Categorical Variable: Value}
    dcc.Store(id='store-auto-data', data = {}), # Dictionary of AutoBenchmarking Model Data {Train Dataset: Value, Test Dataset: Value, Other Dataset: Value, Metric: Value, Auto Feat Selection: Yes/No}
])


# Callback
# Save Project Config and Show Input Forms
@app.callback(
    [Output('store-project','data'),Output('user-input','children')],
    [Input("project-name", "value"), Input("model-dropdown", "value"),
    Input('submit-button','n_clicks')]
    )

def generate_hyperparams(project_name, algo_value, n_clicks):
    project_config = {'Project Name': project_name, 'Algorithm': algo_value}
    ctx = dash.callback_context
    if not ctx.triggered:
        return [], None
    trigger_id = ctx.triggered[0]['prop_id']
    if trigger_id == 'model-dropdown.value':
        return project_config, rep_dataset_layout()
    elif trigger_id == 'submit-button.n_clicks':
        if n_clicks == 1:
            return project_config, auto_dataset_layout()
        if n_clicks == 2:
            return project_config, dcc.Location(id='url', pathname='/results') # should be result pages
            
    else:
        return [], None
    
# Define the callback to change the button label
@app.callback(
    Output('submit-button', 'children'),
    [Input('submit-button', 'n_clicks')]
)
def update_button_label(n_clicks):
    if n_clicks == 1:
        return 'Test'
    else:
        return 'OK'
    

# Update File Paths in input text box for Replicating Model
@app.callback(
    [Output('hyperparams-input', 'value'),
    Output('train-dataset-input', 'value'),
    Output('test-dataset-input', 'value'),
    Output('other-dataset-input', 'value')],
    [Input('hyperparams-upload', 'contents'),
    Input('train-dataset-upload', 'contents'),
    Input('test-dataset-upload', 'contents'),
    Input('other-dataset-upload', 'contents')],
    [State('hyperparams-upload', 'filename'),
    State('train-dataset-upload', 'filename'),
    State('test-dataset-upload', 'filename'),
    State('other-dataset-upload', 'filename')]
)
def update_rep_dataset_inputs(hyperparams_contents, train_contents, test_contents, other_contents, hyperparams_filename, train_filename, test_filename, other_filename):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'hyperparams-upload' in changed_id and hyperparams_contents is not None:
        return hyperparams_filename, dash.no_update, dash.no_update, dash.no_update
    elif 'train-dataset-upload' in changed_id and train_contents is not None:
        return dash.no_update, train_filename, dash.no_update, dash.no_update
    elif 'test-dataset-upload' in changed_id and test_contents is not None:
        return dash.no_update, dash.no_update, test_filename, dash.no_update
    elif 'other-dataset-upload' in changed_id and other_contents is not None:
        return dash.no_update, dash.no_update, dash.no_update, other_filename
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    
# Save Data and Configure Input Variables (For Replicating)
@app.callback(
    Output('store-rep-data', 'data'),
    Input('submit-button', 'n_clicks'),
    [State('hyperparams-upload', 'contents'), State('hyperparams-upload', 'filename'),
    State('train-dataset-upload', 'contents'), State('train-dataset-upload', 'filename'),
    State('test-dataset-upload', 'contents'), State('test-dataset-upload', 'filename'),
    State('other-dataset-upload', 'contents'), State('other-dataset-upload', 'filename'),
    State('target-var-input', 'value'), 
    State('cat-var-input', 'value')],
    prevent_initial_call=True
)
def save_rep_data(n_clicks, hyperparams_contents, hyperparams_filename, train_contents, train_filename, test_contents, test_filename, other_contents, other_filename, target, cat_var):
    if not n_clicks:
        raise PreventUpdate
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'submit-button.n_clicks':
        rep_data = {}
        if hyperparams_contents is not None:
            rep_data['Hyperparams'] = parse_data(hyperparams_contents, hyperparams_filename).to_dict()
        if train_contents is not None:
            rep_data['Train Data'] = parse_data(train_contents, train_filename).to_dict()
        if test_contents is not None:
            rep_data['Test Data'] = parse_data(test_contents, test_filename).to_dict()
        if other_contents is not None:
            rep_data['Other Data'] = parse_data(other_contents, other_filename).to_dict()
        rep_data['Target'] = target
        rep_data['Categorical Var'] = cat_var
        return json.dumps(rep_data)
    else:
        raise PreventUpdate

def parse_data(content, filename):
    """Helper function to parse data in format of csv/xls/txt into pandas dataframe
    Args:
        content (_type_): file content
        filename (_type_): file name
    Returns:
        dat: Dictionary or train/test/other dataframe
    """
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    if 'json' in filename:
        dat = json.loads(decoded)
    if "csv" in filename:
        dat = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    elif "xls" in filename:
        dat = pd.read_excel(io.BytesIO(decoded))
    elif "txt" or "tsv" in filename:
        dat = pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter=r"\s+")
    return dat

# Populate Dropdown list
@app.callback(
    [Output('target-var-input', 'options'), Output('cat-var-input', 'options')],
    [Input('train-dataset-upload', 'contents'), Input('train-dataset-upload', 'filename')]
)
def update_dropdowns(contents, filename):
    if contents is not None:
        # Read the uploaded file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename:
                df = pd.read_excel(io.BytesIO(decoded))
        except Exception as e:
            return [], []
        
        # Process the data to get options for dropdowns
        target_var_options = [{'label': col, 'value': col} for col in df.columns]
        cat_var_options = [{'label': col, 'value': col} for col in df.select_dtypes(include=['object']).columns]
        
        return target_var_options, cat_var_options
    
    return [], []

# Save Data and Configure Input Variables (For Auto-Benchmarking)
@app.callback(
    Output('store-auto-data', 'data'),
    Input('submit-button', 'n_clicks'),
    [State('eval-metric-dropdown', 'value'), 
    State('auto-feat-select-radio', 'value'),
    State('train-auto-upload', 'contents'), State('train-auto-upload', 'filename'),
    State('test-auto-upload', 'contents'), State('test-auto-upload', 'filename'),
    State('other-auto-upload', 'contents'), State('other-auto-upload', 'filename')],
    prevent_initial_call=True
)
def save_auto_data(n_clicks, eval_metric, auto_feat_selection, train_contents, train_filename, test_contents, test_filename, other_contents, other_filename):
    if n_clicks is None:
        raise PreventUpdate
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'submit-button.n_clicks':
        auto_data = {}
        if train_contents is not None:
            auto_data['Train Data'] = parse_data(train_contents, train_filename).to_dict()
        if test_contents is not None:
            auto_data['Test Data'] = parse_data(test_contents, test_filename).to_dict()
        if other_contents is not None:
            auto_data['Other Data'] = parse_data(other_contents, other_filename).to_dict()
        auto_data['Evaluation Metric'] = eval_metric
        auto_data['Feature Selection'] = auto_feat_selection
        return json.dumps(auto_data)
        
    else:
        raise PreventUpdate
    
# Update File Paths in input text box for Auto-Benchmarking Model
@app.callback(
    [Output('train-auto-input', 'value'),
    Output('test-auto-input', 'value'),
    Output('other-auto-input', 'value')],
    [Input('train-auto-upload', 'contents'),
    Input('test-auto-upload', 'contents'),
    Input('other-auto-upload', 'contents')],
    [State('train-auto-upload', 'filename'),
    State('test-auto-upload', 'filename'),
    State('other-auto-upload', 'filename')]
)
def update_auto_dataset_inputs(train_contents, test_contents, other_contents, train_filename, test_filename, other_filename):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'train-auto-upload' in changed_id and train_contents is not None:
        return train_filename, dash.no_update, dash.no_update
    elif 'test-auto-upload' in changed_id and test_contents is not None:
        return dash.no_update, test_filename, dash.no_update
    elif 'other-auto-upload' in changed_id and other_contents is not None:
        return dash.no_update, dash.no_update, other_filename
    else:
        return dash.no_update, dash.no_update, dash.no_update
    
    # callback to update rep-data-output
@app.callback(Output('rep-data-output', 'children'),
              [Input('store-rep-data', 'data')])
def update_rep_data_output(rep_data):
    return json.dumps(rep_data, indent=2)

# callback to update auto-data-output
@app.callback(Output('auto-data-output', 'children'),
              [Input('store-auto-data', 'data')])
def update_auto_data_output(auto_data):
    return json.dumps(auto_data, indent=2)