from auto_ml_validation.app.index import app
from dash.dependencies import Input, Output
from dash import html, dcc
from ..pages.results import *
import pandas as pd

from ...validation_package.algorithms.logistic_regression import LogisticClassifier
from ...validation_package.evaluation.performance_metrics_evaluator import PerformanceEvaluator

############## local variable ##################
project_name = 'Credit Risk'
algo = "Logistic Regression"
sample_size=1000

train = pd.read_csv("~/Desktop/Capstone/loanstats_train_processed.csv")
test = pd.read_csv("~/Desktop/Capstone/loanstats_test_processed.csv")
train_y = train['loan_status']
train_X = train.drop(columns=['loan_status'])
model = LogisticClassifier()
model.fit(train_X, train_y)

test_X = test.drop(columns=['loan_status'])
test_y = test['loan_status']
y_pred = model.predict(test_X)
proba = model.predict_proba(test_X)


# layout
results_layout = html.Div(children=[
    


])

# callbacks
# @app.callback(
#     [Output('threshold-store', 'data')],
#     [Input('threshold', 'value')]
# )
# def store_threshold(threshold):
#     return threshold


@app.callback(
    [Output('dist-curve', 'figure'), Output('roc-curve', 'figure'),
     Output('pr-curve', 'figure'), Output('metrics', 'children')
     ],
    [Input('threshold', 'value')]
)
def generate_performance_metrics(threshold):
    pme = PerformanceEvaluator(proba, float(threshold), test_y, test_X, model._model)

    dist = pme.get_dist_plot()
    roc = pme.get_roc_curve()
    pr = pme.get_pr_curve()
    # lift = pme.get_lift_chart()
    metrics = pme.cal_metrics()


    metrics_comp = html.Div([
        html.H6(f'Accuracy: {metrics["accuracy"]}'),
        html.H6(f'Precision: {metrics["precision"]}'),
        html.H6(f'Recall: {metrics["recall"]}'),
        html.H6(f'F1-Score: {metrics["f1_score"]}'),
        ])

    return dist, roc, pr, metrics_comp