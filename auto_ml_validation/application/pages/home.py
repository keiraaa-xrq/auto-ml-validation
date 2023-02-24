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

def params_layout(hyperparams) -> html.Div:
    """This function renders the hyperparameters form page based on algorithm selected. This is for replicating the model.

    Args:
        hyperparams (list): hyperparameters to tune/train (replicate) on

    Returns:
        html.Div: html element consisting of form input for user
    """
    inputs = []
    for param in hyperparams:
        inputs.append(html.Div([
            html.Label(f"{param.capitalize()}:  "),
            dcc.Input(id=f"{param}-input", type='text', value='', style={'display': 'flex', 'flex-direction': 'column', "margin": "auto"}),
        ]))
    
    return html.Div(id='hyperparams-input', children=[
        html.Div([
            html.P("Please input model hyperparameters:")
        ] + inputs, 
            style=container_style
        )
        ])
    
def rep_dataset_layout() -> html.Div:
    """This function renders the datasets form input page together with the fields (Target and Categorical Variable) after the user clicks OK (For Replicating the Model)

    Returns:
        html.Div: html element consisting of form input for user
    """

    return html.Div([
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
            html.H6('Validation Dataset', style={'textAlign': 'center'}),
            html.Div([
            dcc.Input(id='val-dataset-input', placeholder = 'Select Validation Dataset...', type='text', value=''),
            dcc.Upload(
                id='val-dataset-upload',
                children=html.Div([
                    html.A('Browse File', style = {'color':'black'})
                ]),
                style=upload_style,
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
                    html.A('Browse File',style = {'color':'black'})
                ]),
                style= upload_style,
              ),
            ], style = {'display':'flex','alignItems':'center','justifyContent': 'center', 'gap': '10px'}),
            ]),
            # Text Fields
            html.Div([
            html.P('Please indicate the target variable name:'),
            dcc.Input(id='target-var-input', type='text', value='', style={'width': '100%'}),
            html.P('Please indicate all categorical variables:'),
            dcc.Input(id='cat-var-input', type='text', value='', style={'width': '100%'})
            ], style={'textAlign': 'center', 'margin': 'auto', 'maxWidth': '800px', 'paddingTop': '50px'})
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
            # Instructions
            # Dropdown
            html.H6('Please select evaluation metric:', style={'textAlign': 'center'}),
            dcc.Dropdown(
                    id='eval-metric-dropdown',
                    options=[
                        {'label': 'F1', 'value': 'f1'},
                        {'label': 'Precision', 'value': 'precision'},
                        {'label': 'Accuracy', 'value': 'accuracy'},
                        {'label': 'Recall', 'value': 'recall'}
                    ],
                    value='f1'
                ),
            html.H6('Incorporate Auto-Feature Selection?', style={'textAlign': 'center'}),
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
            html.H5('Please upload the processed datasets for auto benchmarking:', style={'textAlign': 'center'}),
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
            html.H6('Validation Dataset', style={'textAlign': 'center'}),
            html.Div([
            dcc.Input(id='val-auto-input', placeholder = 'Select Validation Dataset...', type='text', value=''),
            dcc.Upload(
                id='val-auto-upload',
                children=html.Div([
                    html.A('Browse File', style = {'color':'black'})
                ]),
                style=upload_style,
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
                    html.A('Browse File',style = {'color':'black'})
                ]),
                style= upload_style,
              ),
            ], style = {'display':'flex','alignItems':'center','justifyContent': 'center', 'gap': '10px'}),
            ]),
            ],
        style=container_style
        )
