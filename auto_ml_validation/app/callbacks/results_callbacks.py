from auto_ml_validation.app.index import app
from auto_ml_validation.app.pages.results import *
from dash.dependencies import Input, Output, State
import json
# Body Content


# Layout
results_layout = html.Div([
    results_p_layout,
])

@app.callback(Output('output-div', 'children'),
              [Input('validator-input-trigger', 'data'),
               Input('validator-input-file','data')]
)
def another_callback(trigger, input_file):
    if trigger:
        print(input_file)
        return 'Callback triggered!'
    return ''

