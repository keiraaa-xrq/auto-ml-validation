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
              [Input('validator-input-trigger', 'data')])
def another_callback(trigger):
    if trigger:
        return 'Callback triggered!'
    return ''

