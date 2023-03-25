import dash
from dash import html, dcc, dash_table
import pandas as pd
from auto_ml_validation.validation_package.evaluation import statistical_metrics_evaluator



############## start of codes used to load dummy data, remove later #######################
train_prob = pd.read_csv('././data/stage_2/loanstats_train_proba.csv')
train_processed = pd.read_csv('././data/stage_2/loanstats_train_processed.csv')
test_prob = pd.read_csv('././data/stage_2/loanstats_test_proba.csv')
test_processed = pd.read_csv('././data/stage_2/loanstats_test_processed.csv')
# prepare the processed dataset
train_processed['probability'] = train_prob['probability']
test_processed['probability'] = test_prob['probability']

test_set = pd.read_csv('././data/stage_2/loanstats_2019Q1_test.csv', index_col = False)
train_set = pd.read_csv('././data/stage_2/loanstats_2019Q1_train.csv', index_col = False)
############## end of codes used to load dummy data, remove later #######################

def statistical_model_metrics_layout(train_set: pd.DataFrame,
               test_set: pd.DataFrame,
               processed_train: pd.DataFrame,
               processed_test: pd.DataFrame,
               prob_col: str, 
               number_of_bins: int) -> html.Div:
    
    my_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(train_set, 
                                           test_set, 
                                           processed_train,
                                           processed_test)
    
    psi_score, psi_df = my_class.calculate_psi(prob_col, number_of_bins)
    # print(psi_df)
    print(psi_df.index.value)
    ks_score = my_class.kstest(prob_col)

    if ks_score.pvalue <= 0.05:
        ks_string = str(ks_score.pvalue) + '. ' + ' Null hypothese (Training and Testing Set are from the same statistical distribution) is rejected.'
    else:
        ks_string = str(ks_score.pvalue) + '. ' + ' Null hypothese (Training and Testing Set are from the same statistical distribution) cannot be rejected.'
    return html.Div(style={'backgroundColor':'#fee6c8', 'width': '95%', 'margin': 'auto'}, children = [
        html.Div([html.H3('Probability Stability Index Table', style={'textAlign': 'center', 'fontWeight': 'bold'}),
                  html.H5('PSI Score: '+str(psi_score), style={'textAlign': 'center', 'fontWeight': 'bold'}),], 
                  style={'width': '100%', 'top': 0, 'left': 0, 'margin': 0}),
        dash_table.DataTable(id="psi-table",
                         data= psi_df.to_dict('records'), 
                         columns=[{"name": i, "id": i} for i in psi_df.columns],
                         sort_action='native',
                         ),
        html.Br(),
        html.H6('P-value of KS Test: ' + ks_string, style={'textAlign': 'left', 'fontWeight': 'bold'}),
        html.Br(),
    ]
    )


def feature_metrics_layout(train_set: pd.DataFrame,
                           test_set: pd.DataFrame,
                            processed_train: pd.DataFrame,
                            processed_test: pd.DataFrame,
                            ft_name_list: [str],
                            number_of_bins: int)->html.Div:
    
    stats_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(
                                           train_set, 
                                           test_set, 
                                           processed_train,
                                           processed_test)
    
    csi_df, csi_dict = stats_class.csi_for_all_features(ft_name_list, number_of_bins)
    
    return html.Div(style={'backgroundColor': '#d7d7d7', 'width': '95%', 'margin': 'auto'}, children=[
        html.Br(),
        html.H3('Feature Metrics', style={'textAlign': 'left', 'fontWeight': 'bold'}),
        #html.Br(),
        html.Label('Please choose the feature to be displayed: '),
        dcc.Dropdown(ft_name_list, ft_name_list[0], 
                     style={'width': '50%', 'margin': 'left'}),
        html.Br(),
        html.H4('GINI Index: ' + str(0.71), style={'textAlign': 'left', 'fontWeight': 'bold'}),
        html.Div([
                    html.H4('Characteristic Stability Index Table', style={'textAlign': 'center', 'fontWeight': 'bold'}),
                    html.H5('CSI Score:    '+ str(csi_dict['loan_amnt']), style={'textAlign': 'center', 'fontWeight': 'bold'}),
        ], style={'width': '100%', 'top': 0, 'left': 0, 'margin': 0}),
        dash_table.DataTable(id="csi-table",
                    data= csi_df[0].to_dict('records'), 
                    columns=[{"name": i, "id": i} for i in csi_df[0].columns],
                    sort_action='native',
                    #style={'width': '100%'}
                    ),
        html.Br(),
        html.H4('Partial Dependency Plot', style={'textAlign': 'center', 'fontWeight': 'bold'}),
        # html.Img(src = "assets/images/PartialDependencyPlot.jpg", 
        #          alt="PartialDependencyPlot",
        #          className = 'center',
        #          style={'width': '70%', 'margin': 'right', 'height': '70%'}),
        html.Br()
        
    ]) 


    

results_layout = html.Div(children=[
    html.Br(),
    html.Br(),
    html.Br(),
    statistical_model_metrics_layout(train_set, test_set, train_processed, test_processed, 'probability', 10),
    html.Br(),
    html.Br(),
    feature_metrics_layout(train_set, test_set, train_processed, test_processed, 
                           ['loan_amnt', 'int_rate','installment','annual_inc'], 10)
    ])

