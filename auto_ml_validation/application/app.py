import pandas as pd
import io
from dash import Dash, html, dcc
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import base64

from pages.home import params_layout, rep_dataset_layout, auto_dataset_layout

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED])
app.title = 'Model Validation Tool'

##################### Variables Config #####################
# Map algorithms to their hyperparameters
algorithm_params = {
    "LR": ['penalty', 'C', 'solver', 'max_iter'],
    "DT": ['criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf'],
    "KNN": ['n_neighbours', 'weights', 'algorithm', 'leaf_size'],
    "RF": ['n_estimators', 'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes'],
    "SVM": ['kernel', 'degree', 'gamma', 'max_iter'],
    "XGB": ['eta','gamma','max_depth','min_child_weight','lambda','alpha']
}

########################## Navbar ##########################
# Input
## None


# Output
logo = "images/MariBank.png"

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(html.Img(src=app.get_asset_url("images/MariBank.png"), height="200px", width = '250px')),
                    dbc.Col(html.H2(app.title, style={"color": "#ec6b12", "font-weight":"bold"}, className="navbar-brand")),
                ],
                align="center"
            )
        ]
    ),
    color="white",
    dark=False,
    sticky="top",
    style={
        "display": "flex",
        "justify-content": "center",
        "align-items": "center",
        "height": "10vh",
    }
)

########################## Body ##########################
# Input
# None



# Output
form_fields = [dbc.Row([
        dbc.Col(
            dbc.Form([
                dbc.Label("Enter Project Name"),
                dbc.Input(id="project-name", placeholder="Project Name", type="text")
            ]),
            width={"size": 3, "offset": 4},
            className="text-center"
        )
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col(
            dbc.Form([
                dbc.Label("Select Model's Algorithm"),
                dcc.Dropdown(
                    id="model-dropdown",
                    options=[
                        {"label": "Logistic Regression", "value": 'LR'},
                        {"label": "Decision Tree", "value": 'DT'},
                        {"label": "Support Vector Machine", "value": 'SVM'},
                        {'label': "Random Forest", "value": 'RF'},
                        {'label': "K-Nearest Neighbours", "value": 'KNN'},
                        {"label": "XGBoost", "value": 'XGB'}
                    ],
                    value=None
                )
            ]),
            width={"size": 3, "offset": 4},
            className="text-center"
        )
    ])
    ,
# Variables Storing
    dcc.Store(id='store-project', data={}, storage_type='memory'), # Dictionary of Project Config {project name: value, algorithm: value}
    dcc.Store(id='hyperparams-value', data = {}), # Dictionary of Hyperparameters {hyperparameter: value}
    dcc.Store(id='store-rep-data', data = {}),
    html.Div(id='user-input')
    ]

form = dbc.Form(form_fields, id="form")

submit_button = html.Div(dbc.Button("OK", 
                                    id="submit-button", 
                                    color="primary",
                                    className="mx-auto d-block"),
                         style={"marginTop": "20px", "textAlign": "center"})


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
    hyperparams = algorithm_params.get(algo_value)
    if trigger_id == 'model-dropdown.value':
        return project_config, params_layout(hyperparams)
    elif trigger_id == 'submit-button.n_clicks':
        if n_clicks == 1:
            return project_config, rep_dataset_layout()
        if n_clicks == 2:
            return project_config, auto_dataset_layout()
            
    else:
        return [], None

# Store hyperparameters Input After OK
@app.callback(
            [Output("hyperparams-value", "data")],
            Input("submit-button", "n_clicks"),
            [State("hyperparams-input",'children')]
            )

def save_hyperparams(n_clicks, user_input):
    input_dict = {}
    if n_clicks is not None and n_clicks > 0:
        elements = user_input[0]['props']['children']
        for input_box in elements[1:]: # First item is string element
            sliced = input_box['props']
            hyperparam = sliced['children'][1]['props']['id'].split('-')[0]
            value = sliced['children'][1]['props']['value']
            input_dict[hyperparam] = value
    print(input_dict)
    return [input_dict]

# Update File Paths in input text box
@app.callback(
    [Output('train-dataset-input', 'value'),
     Output('val-dataset-input', 'value'),
     Output('test-dataset-input', 'value')],
    [Input('train-dataset-upload', 'contents'),
     Input('val-dataset-upload', 'contents'),
     Input('test-dataset-upload', 'contents')],
    [State('train-dataset-upload', 'filename'),
     State('val-dataset-upload', 'filename'),
     State('test-dataset-upload', 'filename')]
)
def update_dataset_inputs(train_contents, val_contents, test_contents, train_filename, val_filename, test_filename):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'train-dataset-upload' in changed_id and train_contents is not None:
        return train_filename, dash.no_update, dash.no_update
    elif 'val-dataset-upload' in changed_id and val_contents is not None:
        return dash.no_update, val_filename, dash.no_update
    elif 'test-dataset-upload' in changed_id and test_contents is not None:
        return dash.no_update, dash.no_update, test_filename
    else:
        return dash.no_update, dash.no_update, dash.no_update
    
# Save Data and Configure Input Variables
@app.callback(
    Output('store-rep-data', 'data'),
    Input('submit-button', 'n_clicks'),
    [State('train-dataset-upload', 'contents'), State('train-dataset-upload', 'filename'),
     State('val-dataset-upload', 'contents'), State('val-dataset-upload', 'filename'),
     State('test-dataset-upload', 'contents'), State('test-dataset-upload', 'filename'),
     State('target-var-input', 'value'), 
     State('cat-var-input', 'value')],
     prevent_initial_call=True
)
def save_rep_data(n_clicks, train_contents, train_filename, val_contents, val_filename, test_contents, test_filename, target, cat_var):
    if not n_clicks:
        raise PreventUpdate
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'submit-button.n_clicks':
        rep_data = {}
        if train_contents is not None:
            rep_data['Train Data'] = parse_data(train_contents, train_filename)
        if val_contents is not None:
            rep_data['Val Data'] = parse_data(val_contents, val_filename)
        if test_contents is not None:
            rep_data['Test Data'] = parse_data(test_contents, test_filename)
        rep_data['Target'] = target
        rep_data['Categorical Var'] = cat_var
        return [rep_data]
    else:
        raise PreventUpdate

def parse_data(content, filename) -> pd.DataFrame:
    """Helper function to parse data in format of csv/xls/txt into pandas dataframe

    Args:
        content (_type_): file content
        filename (_type_): file name
    Returns:
        pd.DataFrame: train/val/test dataframe
    """
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    if "csv" in filename:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    elif "xls" in filename:
        df = pd.read_excel(io.BytesIO(decoded))
    elif "txt" or "tsv" in filename:
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter=r"\s+")
    return df
        
########################## Layout ##########################
app.layout = html.Div([ 
    navbar,
    form,
    submit_button
])

if __name__ == '__main__':
	app.run_server(debug=False)
 