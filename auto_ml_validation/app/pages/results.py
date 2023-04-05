import dash
from dash import html, dcc, dash_table
import pandas as pd
import numpy as np
from auto_ml_validation.validation_package.evaluation import statistical_metrics_evaluator
from auto_ml_validation.validation_package.evaluation import transparency_metrics_evaluator
from auto_ml_validation.validation_package.evaluation import performance_metrics_evaluator
import pickle
from sklearn.linear_model import LogisticRegression
from auto_ml_validation.app.index import app
import matplotlib.pyplot as plt


with open('././data/validator_input/featselection_rf_2023-03-28_data.pkl', 'rb') as f:
    data = pickle.load(f)
# print(data)

bm_train_data = data['bm_train_data']
bm_test_data = data['bm_other_data']['Test']
re_train_data = data['re_train_data']
re_test_data = data['re_other_data']['Test']

def bm_performance_metric_layout():

    return html.Div(style={
        'border': '2px solid black',
        'padding': '20px',
        'background-color': '#FFFBFA',
        'width': '60%',
        'margin': 'auto',
        'text-align': 'center'
    }, children=[
        html.H2('Model-Based Metrics', style={'font-weight': 'bold'}),
        html.Div(style={'display': 'flex', 'justify-content': 'space-between'}, children=[
            dcc.Graph(id='dist-curve1', figure={})]),
        html.Div(style={'display1': 'flex', 'justify-content': 'space-between'}, children=[
            dcc.Graph(id='roc-curve1', figure={})]),
        html.Div(style={'display': 'flex', 'justify-content': 'space-between'}, children=[
            dcc.Graph(id='pr-curve1', figure={})]),
        html.H6('Adjust the threshold here:', style={'font-weight': 'bold'}),
        dcc.Input(id='threshold1', type='range', value=0.5, min=0, max=1),
        html.Div([],
                 style={'display': 'flex', 'justify-content': 'space-between', 'text-align': 'center'},
                 id='metrics1',)
                 
        ])

def re_performance_metric_layout():

    return html.Div(style={
        'border': '2px solid black',
        'padding': '20px',
        'background-color': '#FFFBFA',
        'width': '60%',
        'margin': 'auto',
        'text-align': 'center'
    }, children=[ 
        html.H2('Model-Based Metrics', style={'font-weight': 'bold'}),
        
        html.Div(style={'display': 'flex', 'justify-content': 'space-between'}, children=[
            dcc.Graph(id='dist-curve2', figure={})]),
        html.Div(style={'display': 'flex', 'justify-content': 'space-between'}, children=[
            dcc.Graph(id='roc-curve2', figure={})]),
        html.Div(style={'display': 'flex', 'justify-content': 'space-between'}, children=[
            dcc.Graph(id='pr-curve2', figure={})]),
        html.H6('Adjust the threshold here:', style={'font-weight': 'bold'}),
        dcc.Input(id='threshold2', type='range', value=0.5, min=0, max=1),
        html.Div([],
                 style={'display': 'flex', 'justify-content': 'space-between', 'text-align': 'center'},
                 id='metrics2',)
        ])

def bm_statistical_model_metrics_layout(train_data: pd.DataFrame,
               test_data: pd.DataFrame,
               number_of_bins: int) -> html.Div:
    
    my_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(train_data, 
                                                                         test_data)
    psi_score, psi_df = my_class.calculate_psi(number_of_bins)
    psi_df.columns = psi_df.columns.astype(str)
    psi_df = psi_df.reset_index()
    psi_df['index'] = psi_df['index'].astype(str)
    psi_df.rename(columns = {'index':'ranges'}, inplace = True)
    psi_df = psi_df.round(2)
    ks_dict = my_class.kstest()


    return html.Div(style={
        'border': '2px solid black',
        'background-color': '#FFFBFA',
        'width': '60%',
        'margin': 'auto',
        'text-align': 'center',
        'padding': '20px'
    }, children=[
        html.H3('Probability Stability Index Table', style={'font-weight': 'bold'}),
        html.H5(id='psi-score1',
                children='PSI Score: ' + str(psi_score),
                style={'font-weight': 'bold'}),
        html.Br(),
        html.Label('Please choose the number of bins to be sliced: '),
        dcc.Input(id='psi-num-of-bins1',
                  type='number',
                  value=10,
                  min=2,
                  max=50,
                  step=1,
                  pattern=r'\d*'),
        html.Br(),
        dash_table.DataTable(id='psi-table1',
                              data=psi_df.to_dict('records'),
                              columns=[{'name': i, 'id': i} for i in psi_df.columns],
                              sort_action='native'),
        html.Br(),
        html.H6('KS Test Train: ' + str(ks_dict['Train']), style={'font-weight': 'bold'}),
        html.H6('KS Test Test: ' + str(ks_dict['Test']), style={'font-weight': 'bold'}),
        html.H6('KS Test Train&Test: ' + str(ks_dict['Train vs Test']), style={'font-weight': 'bold'}),
        html.Br(),
    ])

def re_statistical_model_metrics_layout(train_data: pd.DataFrame,
               test_data: pd.DataFrame,
               number_of_bins: int) -> html.Div:
    
    my_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(train_data, 
                                                                         test_data)
    psi_score, psi_df = my_class.calculate_psi(number_of_bins)
    psi_df.columns = psi_df.columns.astype(str)
    psi_df = psi_df.reset_index()
    psi_df['index'] = psi_df['index'].astype(str)
    psi_df.rename(columns = {'index':'ranges'}, inplace = True)
    psi_df = psi_df.round(2)
    ks_dict = my_class.kstest()

    return html.Div(style={
        'border': '2px solid black',
        'background-color': '#FFFBFA',
        'width': '60%',
        'margin': 'auto',
        'text-align': 'center',
        'padding': '20px'
    }, children=[
        html.H3('Probability Stability Index Table', style={'font-weight': 'bold'}),
        html.H5(id='psi-score2',
                children='PSI Score: ' + str(psi_score),
                style={'font-weight': 'bold'}),
        html.Br(),
        html.Label('Please choose the number of bins to be sliced: '),
        dcc.Input(id='psi-num-of-bins2',
                  type='number',
                  value=10,
                  min=2,
                  max=50,
                  step=1,
                  pattern=r'\d*'),
        html.Br(),
        dash_table.DataTable(id='psi-table2',
                              data=psi_df.round(2).to_dict('records'),
                              columns=[{'name': i, 'id': i} for i in psi_df.columns],
                              sort_action='native'),
        html.Br(),
        html.H6('KS Test Train: ' + str(ks_dict['Train']), style={'font-weight': 'bold'}),
        html.H6('KS Test Test: ' + str(ks_dict['Test']), style={'font-weight': 'bold'}),
        html.H6('KS Test Train&Test: ' + str(ks_dict['Train vs Test']), style={'font-weight': 'bold'}),
        html.Br(),
    ])

# define ling-running task
def bm_gini_layout(train_data: pd.DataFrame,
                test_data: pd.DataFrame, 
                ft_name_list: list[str])-> html.Div:
    
    return html.Div(
        style={
            'border': '2px solid black',
            'padding': '20px',
            'width': '60%',
            'margin': 'auto',
            'background-color': '#EFEFEF',
            'text-align': 'center'
        },
        children=[
            html.Br(),
            html.H2('Feature-Based Metrics', style={'font-weight': 'bold'}),
            html.H3('GINI Index',  style={
                'textAlign': 'center', 
                'fontWeight': 'bold',
                'margin-bottom': '20px'  # add margin below title
            }),
            html.Label('Select Features to Display:', style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id="gini-feature-multi-dynamic-dropdown1", 
                multi=True,
                options=dict(zip(train_data['raw_X'].columns.to_list(), 
                                  train_data['raw_X'].columns.to_list())),
                value=[],
                style={'width': '50%', 'margin': 'auto'}
            ), 
            html.Br(),
            html.Br(),
            dcc.Loading(
                id="loading-gini1",
                type="circle",
                children=html.Div([
                    # Output of long-running task
                    html.Div(id="gini_viz1", style={'fontWeight': 'bold'}),
                ]),
            ),
            html.Br(),
            html.Br()        
        ]
    )

# define ling-running task
def re_gini_layout(train_data: pd.DataFrame,
                test_data: pd.DataFrame, 
                ft_name_list: list[str])-> html.Div:
    
    return html.Div(
        style={
            'border': '2px solid black',
            'padding': '20px',
            'width': '60%',
            'margin': 'auto',
            'background-color': '#EFEFEF',
            'text-align': 'center'
        },
        children=[
            html.Br(),
            html.H2('Feature-Based Metrics', style={'font-weight': 'bold'}),
            html.H3('GINI Index',  style={
                'textAlign': 'center', 
                'fontWeight': 'bold',
                'margin-bottom': '20px'  # add margin below title
            }),
            html.Label('Select Features to Display:', style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id="gini-feature-multi-dynamic-dropdown2", 
                multi=True,
                options=dict(zip(train_data['raw_X'].columns.to_list(), 
                                  train_data['raw_X'].columns.to_list())),
                value=[],
                style={'width': '50%', 'margin': 'auto'}
            ), 
            html.Br(),
            html.Br(),
            dcc.Loading(
                id="loading-gini2",
                type="circle",
                children=html.Div([
                    # Output of long-running task
                    html.Div(id="gini_viz2", style={'fontWeight': 'bold'}),
                ]),
            ),
            html.Br(),
            html.Br()        
        ]
    )

def bm_csi_table_layout(train_data: pd.DataFrame)->html.Div:
    
    return html.Div(
        style={
            'background-color': '#EFEFEF',
            'border': '2px solid black',
            'padding': '20px',
            'width': '60%',
            'margin': 'auto',
            'text-align': 'center'
        }, 
        children=[
            html.H3('CSI Metrics', style={
                'textAlign': 'center', 
                'fontWeight': 'bold',
                'margin-bottom': '20px'  # add margin below title
            }),
            html.Br(),
            html.Label('Features to Display:', style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id="csi-feature-multi-dynamic-dropdown1", 
                multi=True,
                options=dict(zip(train_data['raw_X'].columns.to_list(), train_data['raw_X'].columns.to_list())),
                value=[],
                style={'width': '50%', 'margin': 'auto'}
            ),
            html.Br(),
            html.Label('Number of Bins:', style={'font-weight': 'bold'}),
            dcc.Input(
                id="csi-num-of-bins1",
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
                html.H4('Characteristic Stability Index Table', style={'text-align': 'center', 'font-weight': 'bold'})
            ], style={'margin': '0'}),
            html.Div(id="feature_related_viz1", children = []),
            html.Br(),
            html.Br()        
        ]
    )

def re_csi_table_layout(train_data: pd.DataFrame)->html.Div:
    
    return html.Div(
        style={
            'background-color': '#EFEFEF',
            'border': '2px solid black',
            'padding': '20px',
            'width': '60%',
            'margin': 'auto',
            'text-align': 'center'
        }, 
        children=[
            html.H3('CSI Metrics', style={
                'textAlign': 'center', 
                'fontWeight': 'bold',
                'margin-bottom': '20px'  # add margin below title
            }),
            html.Br(),
            html.Label('Features to Display:', style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id="csi-feature-multi-dynamic-dropdown2", 
                multi=True,
                options=dict(zip(train_data['raw_X'].columns.to_list(), train_data['raw_X'].columns.to_list())),
                value=[],
                style={'width': '50%', 'margin': 'auto'}
            ),
            html.Br(),
            html.Label('Number of Bins:', style={'font-weight': 'bold'}),
            dcc.Input(
                id="csi-num-of-bins2",
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
                html.H4('Characteristic Stability Index Table', style={'text-align': 'center', 'font-weight': 'bold'})
            ], style={'margin': '0'}),
            html.Div(id="feature_related_viz2", children = []),
            html.Br(),
            html.Br()        
        ]
    )

def bm_trans_layout(train_data:pd.DataFrame,
                 model_path: str)->html.Div:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # print(train_data['processed_X'])

    evaluator = transparency_metrics_evaluator.TransparencyMetricsEvaluator(model, train_data['processed_X'])

    local_lime_fig, global_lime_fig, local_text_lime, global_text_lime = evaluator.lime_interpretability()
    global_lime_fig.savefig('././auto_ml_validation/app/assets/images/bm_global_lime.png', bbox_inches='tight')
    local_lime_fig.savefig('././auto_ml_validation/app/assets/images/bm_local_lime.png', bbox_inches='tight')

    local_shap_fig, global_shap_fig, local_text_shap, global_text_shap = evaluator.shap_interpretability()
    local_shap_fig.savefig('././auto_ml_validation/app/assets/images/bm_local_shap.png',  bbox_inches='tight')
    global_shap_fig.savefig('././auto_ml_validation/app/assets/images/bm_global_shap.png', bbox_inches='tight')

    # Return layout with adjusted border, centered images, and title
    img_style = {'width': '600px', 'height': '400px', 'object-fit': 'contain'}
    return html.Div(style={
        'border': '1px solid black',
        'padding': '10px',
        'margin': '10px auto',
        'max-width': '800px',
        'background-color': '#ADD2FF',
        'text-align': 'center'
    }, children=[html.H2('Transparency Metrics', style={'text-align': 'center'}),
                 html.H4('Local Lime Plot', style={'fontWeight': 'bold'}),
                 html.Img(src=app.get_asset_url("images/bm_local_lime.png"), style=img_style),
                 html.H4('Global Lime Plot', style={'fontWeight': 'bold'}),
                 html.Img(src=app.get_asset_url("images/bm_global_lime.png"), style=img_style),
                 html.H4('Local Shap Plot', style={'fontWeight': 'bold'}),
                 html.Img(src=app.get_asset_url("images/bm_local_shap.png"), style=img_style),
                 html.H4('Global Shap Plot', style={'fontWeight': 'bold'}),
                 html.Img(src=app.get_asset_url("images/bm_global_shap.png"),style=img_style),
                 ])

def re_trans_layout(train_data:pd.DataFrame,
                 model_path: str)->html.Div:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # print(train_data['processed_X'])

    evaluator = transparency_metrics_evaluator.TransparencyMetricsEvaluator(model, train_data['processed_X'])

    local_lime_fig, global_lime_fig, local_text_lime, global_text_lime = evaluator.lime_interpretability()
    global_lime_fig.savefig('././auto_ml_validation/app/assets/images/re_global_lime.png', bbox_inches='tight')
    local_lime_fig.savefig('././auto_ml_validation/app/assets/images/re_local_lime.png', bbox_inches='tight')

    local_shap_fig, global_shap_fig, local_text_shap, global_text_shap = evaluator.shap_interpretability()
    local_shap_fig.savefig('././auto_ml_validation/app/assets/images/re_local_shap.png',  bbox_inches='tight')
    global_shap_fig.savefig('././auto_ml_validation/app/assets/images/re_global_shap.png', bbox_inches='tight')

    # Return layout with adjusted border, centered images, and title
    img_style = {'width': '600px', 'height': '400px', 'object-fit': 'contain'}
    return html.Div(style={
        'border': '1px solid black',
        'padding': '10px',
        'margin': '10px auto',
        'max-width': '800px',
        'background-color': '#ADD2FF',
        'text-align': 'center'
    }, children=[html.H2('Transparency Metrics', style={'text-align': 'center'}),
                 html.H4('Local Lime Plot', style={'fontWeight': 'bold'}),
                 html.Img(src=app.get_asset_url("images/re_local_lime.png"), style=img_style),
                 html.H4('Global Lime Plot', style={'fontWeight': 'bold'}),
                 html.Img(src=app.get_asset_url("images/re_global_lime.png"), style=img_style),
                 html.H4('Local Shap Plot', style={'fontWeight': 'bold'}),
                 html.Img(src=app.get_asset_url("images/re_local_shap.png"), style=img_style),
                 html.H4('Global Shap Plot', style={'fontWeight': 'bold'}),
                 html.Img(src=app.get_asset_url("images/re_global_shap.png"),style=img_style),
                 ])



















