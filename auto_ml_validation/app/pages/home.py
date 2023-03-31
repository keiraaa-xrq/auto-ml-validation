from dash import html, dcc
import dash_bootstrap_components as dbc

container_style = {
    # For Border Container where we store the input form
            'textAlign': 'center',
            'marginTop': '50px',
            'margin': '20px auto',
            'width': '50%',  # adjust this to control the width of the container/box
            'border': '2px solid orange',  # add border to create a container/box
            'padding': '20px', # add padding for content spacing,
            'backgroundColor': '#feeeea',
            'display': 'block'
            }

upload_style = {
    # For Browse Files dcc.Upload component
                'width': '120px',
                'height': '30px',
                'borderWidth': '1px',
                'borderStyle': 'solid',
                'borderRadius': '5px',
                'textAlign': 'center',
                'cursor': 'pointer',
                'color': 'orange',
                'margin': '0 auto',
                'backgroundColor':'#ff8d74'
            }

def project_field():
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
        html.Div(id="validation-message", style={"color": "red", 'textAlign': 'center'}),
        html.Div(id="failed-modeling-message", style={"color": "red", 'textAlign': 'center'})
        ]

    form = dbc.Form(form_fields, id="form")

    submit_button = html.Div(dbc.Button("Start", 
                                        id="submit-button", 
                                        color="primary",
                                        className="mx-auto d-block"),
                            style={"marginTop": "20px", "textAlign": "center"})
    
    return form, submit_button

    
def rep_dataset_layout() -> html.Div:
    """This function renders the datasets form input page together with the fields (Target and Categorical Variable) after the user clicks OK (For Replicating the Model)
    Returns:
        html.Div: html element consisting of form input for user
    """
    # Create the checklist dropdown
    cat_var_dropdown = dbc.DropdownMenu(
        label="Select Options",
        children=[
            dbc.Checklist(
                options=[],
                value=[],
                id="cat-var-input",
                inline=True,
            ),
        ],
    )
    return html.Div([
        # New Heading
        html.Div([
            html.H5('Part I. Model Replication', style={'textAlign': 'left', 'fontWeight': 'bold'})
        ], style={'backgroundColor': '#ffaf77', 'width': '100%', 'top': 0, 'left': 0, 'margin': 0}),
        dbc.Row([
            dbc.Col(
                dbc.Form([
                    html.H5("Select Model's Algorithm: ", style={'textAlign': 'center'}),
                    dcc.Dropdown(
                        id="model-dropdown",
                        options=[
                            {"label": "Logistic Regression", "value": 'logistic'},
                            {"label": "Decision Tree", "value": 'dt'},
                            {"label": "Support Vector Machine", "value": 'svm'},
                            {'label': "Random Forest", "value": 'rf'},
                            {'label': "K-Nearest Neighbours", "value": 'knn'},
                            {"label": "XGBoost", "value": 'xgboost'}
                        ],
                        value=None
                    )
                ]),
                width={"size": 3, "offset": 4},
                className="text-center",
            )
        ], style={"margin-top": "10px", 'margin-bottom': '30px'}),
        html.Div([
            # Instructions
            html.H5("Please upload the JSON file of model's hyperparameters: ", style={'textAlign': 'center'}),
            html.H6("Within the file, keys will be the parameters name E.g. {class_weight: {0:0.8, 1:0.2}}", style={'textAlign': 'center', 'color': '#4f4f4f ', 'font-size': '13px'}),
            # Files Upload
            html.H6('Hyperparameters Values', style={'textAlign': 'center'}),
            html.Div([
            dcc.Input(id='hyperparams-input', placeholder = 'Select JSON File...', type='text', value=''),
            dcc.Upload(
                id='hyperparams-upload',
                children=html.Div([  
                    html.A('Browse File', style = {'color':'black'})
                ]),
            style= upload_style,
            ),
            ], style = {'display':'flex','alignItems':'center','justifyContent': 'center', 'gap': '10px', 'margin-bottom': '30px'}),
            ]),
        html.Div([
            # Instructions
            html.H5('Please upload the processed datasets for running the model:', style={'textAlign': 'center'}),
            html.H6('Datasets should be processed and contain only selected features along with the target variable.', style={'textAlign': 'center', 'color': '#4f4f4f ', 'font-size': '13px'}),
            # Files Upload
            html.H6('Train Dataset', style={'textAlign': 'center'}),
            html.Div([
            dcc.Input(id='train-dataset-input', placeholder = 'Select Train Dataset...', type='text', value=''),
            dcc.Upload(
                id='train-dataset-upload',
                children=html.Div([  
                    html.A('Browse File', style = {'color':'black'})
                ]),
            style= upload_style,
            ),
            ], style = {'display':'flex','alignItems':'center','justifyContent': 'center', 'gap': '10px'}),
            ]),
            html.Div([
            html.H6('Test Dataset', style={'textAlign': 'center'}),
            html.Div([
            dcc.Input(id='test-dataset-input', placeholder = 'Select Test Dataset...', type='text', value=''),
            dcc.Upload(
                id='test-dataset-upload',
                children=html.Div([
                    html.A('Browse File', style = {'color':'black'})
                ]),
                style=upload_style,
             ),
            ], style = {'display':'flex','alignItems':'center','justifyContent': 'center', 'gap': '10px'}),
            ]),
            html.Div([
            html.H6('Other Dataset', style={'textAlign': 'center'}),
            html.Div([
            dcc.Input(id='other-dataset-input', placeholder = 'Select Other Dataset...', type='text', value=''),
            dcc.Upload(
                id='other-dataset-upload',
                children=html.Div([
                    html.A('Browse File',style = {'color':'black'})
                ]),
                style= upload_style,
              ),
            ], style = {'display':'flex','alignItems':'center','justifyContent': 'center', 'gap': '10px'}),
            ]),
            # Text Fields
            html.Div([
            html.P('Please indicate the target variable name:'),
            dcc.Dropdown(id='target-var-input', value='', options = [], style={'width': '100%'}),
            html.P('Please indicate all categorical variables:'),
            cat_var_dropdown
            ], style={'textAlign': 'center', 'margin': 'auto', 'maxWidth': '800px', 'paddingTop': '50px'}),
            dcc.Textarea(id='selected-options', value='', readOnly=True, style={'margin-top': '20px', 'width': '78%', 'height': '80%'})
            ], 
        style=container_style
        )
  
  
def auto_dataset_layout() -> html.Div:
    """This function renders the datasets form input page together with the fields (Evaluation Metric, Auto-Feat Selection) after the user clicks OK (For Auto-benchmarking)
    Returns:
        html.Div: html element consisting of form input for user
    """

    return html.Div([
        html.Div([
            html.H5('Part II. Automatic Benchmark', style={'textAlign': 'left', 'fontWeight': 'bold'})
        ], style={'backgroundColor': '#ffaf77', 'width': '100%', 'top': 0, 'left': 0, 'margin': 0}),
        html.Div([
            # Instructions
            # Dropdown
            html.H6('Please select evaluation metric:', style={'textAlign': 'center'}),
            dcc.Dropdown(
                    id='eval-metric-dropdown',
                    options=[
                        {'label': 'F1', 'value': 'f1'},
                        {'label': 'Precision', 'value': 'precision'},
                        {'label': 'Accuracy', 'value': 'accuracy'},
                        {'label': 'Recall', 'value': 'recall'},
                        {'label': 'AUC-ROC', 'value': 'roc_auc'}
                    ],
                    value='f1'
                ),
            html.H6('Incorporate Auto-Feature Selection?', style={'textAlign': 'center', 'margin-top': '20px'}),
                dcc.RadioItems(
                    id='auto-feat-select-radio',
                    options=[
                        {'label': 'Yes', 'value': 'yes'},
                        {'label': 'No', 'value': 'no'}
                    ],
                    value='yes',
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                    inputStyle={'margin-right': '5px'},
                    style={'display': 'flex', 'gap': '30px', 'alignItems':'center','justifyContent': 'center'}
                ), 
            html.H5('Please upload the processed datasets for auto benchmarking:', style={'textAlign': 'center','margin-top': '30px'}),
            html.H6('Datasets should be processed and contain only selected features along with the target variable.', style={'textAlign': 'center', 'color': '#4f4f4f ', 'font-size': '13px'}),
            # Files Upload
            html.H6('Train Dataset', style={'textAlign': 'center'}),
            html.Div([
            dcc.Input(id='train-auto-input', placeholder = 'Select Train Dataset...', type='text', value=''),
            dcc.Upload(
                id='train-auto-upload',
                children=html.Div([  
                    html.A('Browse File', style = {'color':'black'})
                ]),
            style= upload_style,
            ),
            ], style = {'display':'flex','alignItems':'center','justifyContent': 'center', 'gap': '10px'}),
            ]),
            html.Div([
            html.H6('Test Dataset', style={'textAlign': 'center'}),
            html.Div([
            dcc.Input(id='test-auto-input', placeholder = 'Select Test Dataset...', type='text', value=''),
            dcc.Upload(
                id='test-auto-upload',
                children=html.Div([
                    html.A('Browse File', style = {'color':'black'})
                ]),
                style=upload_style,
             ),
            ], style = {'display':'flex','alignItems':'center','justifyContent': 'center', 'gap': '10px'}),
            ]),
            html.Div([
            html.H6('Other Dataset', style={'textAlign': 'center'}),
            html.Div([
            dcc.Input(id='other-auto-input', placeholder = 'Select Other Dataset...', type='text', value=''),
            dcc.Upload(
                id='other-auto-upload',
                children=html.Div([
                    html.A('Browse File',style = {'color':'black'})
                ]),
                style= upload_style,
              ),
            ], style = {'display':'flex','alignItems':'center','justifyContent': 'center', 'gap': '10px'}),
            ]),
            ],
        style=container_style
        )
    
def loading_div_layout(app) -> html.Div:
    return html.Div(
            [
                dcc.Loading(
                    id="loading-spinner",
                    children=[
                        html.Div(
                            className="loader",
                            children=[
                                html.Img(
                                    src=app.get_asset_url("images/ball_loading.gif"),
                                    alt="loading...",
                                ),
                                html.H3(
                                    id="loading-text",
                                    className="loader-text",
                                    style={
                                        "textAlign": "center",
                                        "fontWeight": "bold",
                                    },
                                    children="Preparing...",
                                ),
                            ],
                        )
                    ],
                    type="circle",
                    loading_state={"is_loading": True},
                ),
                dcc.Interval(id="interval-component", interval=5000, n_intervals=0),
            ],
            style={"display": "flex", "alignItems": "center", "justifyContent": "center", "gap": "10px"},
        )