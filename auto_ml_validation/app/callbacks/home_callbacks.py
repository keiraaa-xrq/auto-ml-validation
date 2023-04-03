# Landing Page Callbacks
"""
Home page that includes loading layout
"""
from auto_ml_validation.app.index import app
from auto_ml_validation.app.pages.home import *
from auto_ml_validation.validation_package import model_pipeline
from auto_ml_validation.validation_package.utils.utils import instantiate_clf

import pandas as pd
import io
import os
import base64
import json
from datetime import datetime
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# Body Content
form, submit_button = project_field()
rep_layout = rep_dataset_layout()
auto_layout = auto_dataset_layout()
loading_layout = loading_div_layout(app)


# Layout
home_layout = html.Div([
    dcc.Location(id='url', refresh=False, pathname='/home'),
    form,
    html.Div(children=[rep_layout, auto_layout, submit_button], id='content_div'),
    dcc.Store(id='store-project', data={}, storage_type='memory'), # Dictionary of Project Config {project name: value, algorithm: value}
    dcc.Store(id='store-rep-data', data = {}, storage_type ='session'), # Dictionary of Replicating Model Data {Hyperparamaters: Value, Train Dataset: Value, Test Dataset: Value, Other Dataset: Value, Target: Value, Categorical Variable: Value}
    dcc.Store(id='store-auto-data', data = {}, storage_type ='session'), # Dictionary of AutoBenchmarking Model Data {Train Dataset: Value, Test Dataset: Value, Other Dataset: Value, Metric: Value, Auto Feat Selection: Yes/No}
])

# Callback
# Save Project Config and Show Input Forms
@app.callback(
    [Output('store-project','data')],
    [Input("project-name", "value"), Input("model-dropdown", "value")]
    )

def save_proj_dat(project_name, algo_value):
    if project_name == None:
        project_name = ""
    project_config = {'Project Name': project_name, 'Algorithm': algo_value}
    return [json.dumps(project_config)]

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
        target_var_options = cat_var_options = [{'label': col, 'value': col} for col in df.columns]
        
        return target_var_options, cat_var_options
    
    return [], []

# Populate selected categorical variables in textbox
@app.callback(
    Output('selected-options', 'value'),
    [Input('cat-var-input', 'value')]
)
def update_selected_options(selected_options):
    if selected_options is None:
        return ''
    else:
        return ', '.join(selected_options)
    
    
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
    
# Validate form, save data for replication and auto-benchmarking, update content_div with loading spinner when submit_button is clicked
@app.callback(
    Output('content_div', 'children'), Output('validation-message', 'children'),
    Output('store-rep-data', 'data'), Output('store-auto-data', 'data'),
    Input('submit-button', 'n_clicks'),
    [State('hyperparams-upload', 'contents'), State('hyperparams-upload', 'filename'),
    State("model-dropdown", "value"),
    State('train-dataset-upload', 'contents'), State('train-dataset-upload', 'filename'),
    State('train-auto-upload', 'contents'), State('train-auto-upload', 'filename'),
    State('test-dataset-upload', 'contents'), State('test-dataset-upload', 'filename'),
    State('test-auto-upload', 'contents'), State('test-auto-upload', 'filename'),
    State('other-dataset-upload', 'contents'), State('other-dataset-upload', 'filename'),
    State('other-auto-upload', 'contents'), State('other-auto-upload', 'filename'),
    State('target-var-input', 'value'),
    State('cat-var-input', 'value'),
    State('eval-metric-dropdown', 'value'), 
    State('auto-feat-select-radio', 'value')],
    prevent_initial_call=True,
)
def validate_inputs(n_clicks, hyperparams_content, hyperparams_file, algorithm, 
                    train_rep_content, train_rep_file, train_auto_content, train_auto_file, 
                    test_rep_content, test_rep_file, test_auto_content, test_auto_file, 
                    other_rep_content, other_rep_file, other_auto_content, other_auto_file, target_var, cat_var,
                    eval_metric, auto_feat_selection):
    
    # Check if the submit button has been clicked
    if n_clicks is None:
        raise PreventUpdate
    # Check if an algorithm has been selected
    if algorithm is None:
        return [rep_layout, auto_layout, submit_button], 'Please select an option from the algorithm dropdown.', {}, {}
    # Check if all required files have been uploaded
    if not all([train_rep_file, test_rep_file, train_auto_file, test_auto_file]):
        return [rep_layout, auto_layout, submit_button], 'Please upload all the required dataset files.', {}, {}
    # Check if the target variable has been selected for the replication dataset
    if train_rep_file and not target_var:
        return [rep_layout, auto_layout, submit_button], 'Please select the target variable.', {}, {}
    # Try to parse all uploaded dataset files
    try:
        train_rep = parse_data(train_rep_content, train_rep_file) if train_rep_content else None
        train_auto = parse_data(train_auto_content, train_auto_file) if train_auto_content else None
        test_rep = parse_data(test_rep_content, test_rep_file) if test_rep_content else None
        test_auto = parse_data(test_auto_content, test_auto_file) if test_auto_content else None
        other_rep = parse_data(other_rep_content, other_rep_file) if other_rep_content else None
        other_auto = parse_data(other_auto_content, other_auto_file) if other_auto_content else None
    except Exception as e:
        # Return an error message if any of the files could not be parsed
        file_type = 'training replication' if train_rep_content else 'auto benchmarking training' if train_auto_content else 'testing replication' if test_rep_content else 'auto benchmarking testing' if test_auto_content else 'other dataset replication' if other_rep_content else 'auto benchmarking for other dataset'
        return [rep_layout, auto_layout, submit_button], f'Please upload the correct file for {file_type}. Accepted formats are csv, xls, txt, tsv.', {}, {}
    # Check if hyperparameters have been provided
    if hyperparams_content:
        try:
            # Try to parse hyperparameters and instantiate the model
            params = parse_data(hyperparams_content, hyperparams_file)
            clf = instantiate_clf(algorithm, params)
        except Exception:
            # Return an error message if the hyperparameters could not be parsed or the model could not be instantiated
            return [rep_layout, auto_layout, submit_button], 'Please upload the correct hyperparameters for the model in a JSON file.', {}, {}
    # If all checks pass, populate dcc.Store and return the loading layout
    rep_data = auto_data = {}
    rep_data['Hyperparams'] = params
    rep_data['Train Data'] = train_rep.to_dict()
    rep_data['Test Data'] = test_rep.to_dict()
    if other_rep_content:
        rep_data['Other Data'] = other_rep.to_dict()
    rep_data['Target'] = target_var
    rep_data['Categorical Var'] = cat_var
    
    auto_data['Train Data'] = train_auto.to_dict()
    auto_data['Test Data'] = test_auto.to_dict()
    if other_auto_content:
        auto_data['Other Data'] = other_auto.to_dict()
    auto_data['Evaluation Metric'] = eval_metric
    auto_data['Feature Selection'] = auto_feat_selection
    return loading_layout, None, [json.dumps(rep_data)],  [json.dumps(auto_data)]

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
    elif "csv" in filename:
        dat = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    elif "xls" in filename:
        dat = pd.read_excel(io.BytesIO(decoded))
    elif "txt" or "tsv" in filename:
        dat = pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter=r"\s+")
    return dat

# Begin modeling process when submit_button is clicked   
@app.callback(
    Output("url", "pathname"), Output("failed-modeling-message", "children"), 
    Output("validator-input-trigger", "data"), Output("validator-input-file", "data"),
    [Input("loading-spinner", "loading_state"),
     Input("store-project", "data"),
     Input("store-rep-data", "data")],
    [State("store-auto-data", "data"),
     State('url', 'pathname')],
    prevent_initial_call=True
)
def modelling_process(loading, proj, rep_data, auto_data, current_pathname):
    if loading:
        project_dict = json.loads(proj)
        rep_dict = json.loads(rep_data[0])
        auto_dict = json.loads(auto_data[0])

        project_name = project_dict['Project Name']
        algorithm = project_dict['Algorithm']
        hyperparams = rep_dict['Hyperparams']
        rep_train = pd.DataFrame(rep_dict['Train Data'])
        rep_test = pd.DataFrame(rep_dict['Test Data'])
        rep_other = [pd.DataFrame(rep_dict['Other Data'])] if 'Other Data' in rep_dict else []
        target = rep_dict['Target'].lower()
        cat_cols = [col.lower() for col in rep_dict['Categorical Var']]
        auto_train = pd.DataFrame(auto_dict['Train Data'])
        auto_test = pd.DataFrame(auto_dict['Test Data'])
        auto_other = [pd.DataFrame(auto_dict['Other Data'])] if 'Other Data' in auto_dict else []
        metric = auto_dict['Evaluation Metric']
        bool_map = {"yes": True, "no": False}
        feat_sel = bool_map.get(auto_dict['Feature Selection'])
        try:
            output, file_name = model_pipeline.autoML(project_name, algorithm, hyperparams, 
                                       rep_train, rep_test, rep_other, target, cat_cols, 
                                       auto_train, auto_test, auto_other, metric, feat_sel)
        except Exception as e:
            return '/home', f"Model Building has failed. Error: {e}. Please try again. ", False, None

        return '/results', "", True, file_name
    
    return current_pathname, False, None

# Periodically update progress text
@app.callback(
    Output("loading-text", "children"), 
    [Input("interval-component", "n_intervals"),
    Input("store-project", "data")],
)
def update_loading_text(n_intervals, project_data):
    project_dict = json.loads(project_data)
    project_name = project_dict['Project Name']
    date = datetime.today().strftime('%Y-%m-%d')
    file_path = f"logs/{date}_{project_name}.log"
    if not os.path.exists(file_path):
        return "Preparing..."
    with open(file_path, "r") as f:
        logger_contents = f.read()

    filtered_contents = []
    for line in logger_contents.strip().split("\n"):
        if "main - INFO -" in line:
            filtered_contents.append(line)

    loading_text = filtered_contents[-1].split("-")[-1].strip() if filtered_contents else "Preparing..."
    return loading_text

# Change interval timing based on last loading text
@app.callback(
    Output("interval-component", "interval"),
    [Input("loading-text", "children")]
)
def update_interval(loading_text):
    if loading_text in ["Preparing...", "Almost done...", "Replicating the model..."]:
        return 1000
    else:
        return 30000

