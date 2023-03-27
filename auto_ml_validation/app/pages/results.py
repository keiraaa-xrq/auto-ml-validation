import dash
from dash import html, dcc, dash_table
import pandas as pd
import numpy as np
from auto_ml_validation.validation_package.evaluation import statistical_metrics_evaluator



############## start of codes used to load dummy data, remove later #######################
train_prob = pd.read_csv('././data/stage_2/loanstats_train_proba.csv')
train_processed = pd.read_csv('././data/stage_2/loanstats_train_processed.csv')
test_prob = pd.read_csv('././data/stage_2/loanstats_test_proba.csv')
test_processed = pd.read_csv('././data/stage_2/loanstats_test_processed.csv')

# load the unprocessed dataset
test_set = pd.read_csv('././data/stage_2/loanstats_2019Q1_test.csv', index_col = False)
train_set = pd.read_csv('././data/stage_2/loanstats_2019Q1_train.csv', index_col = False)

all_feature_list = ['loan_amnt', 'int_rate', 'installment', 'home_ownership', 'annual_inc',
       'verification_status','dti', 'delinq_2yrs',
       'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc',
       'initial_list_status', 'out_prncp', 'out_prncp_inv', 'total_pymnt',
       'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
       'total_rec_late_fee', 'last_pymnt_amnt', 'collections_12_mths_ex_med',
       'application_type', 'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m',
       'open_act_il', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il',
       'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc',
       'all_util', 'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m',
       'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util',
       'delinq_amnt', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op',
       'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc',
       'mths_since_recent_bc', 'mths_since_recent_inq',
       'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
       'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
       'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
       'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq',
       'percent_bc_gt_75', 'pub_rec_bankruptcies', 'tax_liens',
       'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit',
       'total_il_high_credit_limit', 'hardship_flag', 'debt_settlement_flag']

train_data, test_data = {}, {}

train_data['raw_X'] = train_set[all_feature_list]
# print(train_data['raw_X'].columns)
# Index(['loan_amnt', 'int_rate', 'installment', 'home_ownership', 'annual_inc',
#        'verification_status', 'dti', 'delinq_2yrs', 'inq_last_6mths',
#        'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'initial_list_status',
#        'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
#        'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
#        'last_pymnt_amnt', 'collections_12_mths_ex_med', 'application_type',
    #    'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_act_il',
    #    'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il',
    #    'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
    #    'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m',
    #    'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util',
    #    'delinq_amnt', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op',
    #    'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc',
    #    'mths_since_recent_bc', 'mths_since_recent_inq',
    #    'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
    #    'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
    #    'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
    #    'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq',
    #    'percent_bc_gt_75', 'pub_rec_bankruptcies', 'tax_liens',
    #    'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit',
    #    'total_il_high_credit_limit', 'hardship_flag', 'debt_settlement_flag']
train_data['processed_X'] = train_processed.drop(['loan_status'], axis=1)
train_data['y'] = train_processed['loan_status']
train_data['pred_proba'] = np.array([1-train_prob['probability'], train_prob['probability']])

test_data['raw_X'] = test_set[all_feature_list]
test_data['processed_X'] = test_processed.drop(['loan_status'], axis=1)
test_data['y'] = test_processed['loan_status']
test_data['pred_proba'] = np.array([1-test_prob['probability'], test_prob['probability']])
############## end of codes used to load dummy data, remove later #######################

def statistical_model_metrics_layout(train_data: pd.DataFrame,
               test_data: pd.DataFrame,
               number_of_bins: int) -> html.Div:
    
    my_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(train_data, 
                                                                         test_data)
    psi_score, psi_df = my_class.calculate_psi(number_of_bins)
    print(psi_df)
    # reset index as columns for display
    psi_df.columns = psi_df.columns.astype(str)
    psi_df = psi_df.reset_index()
    psi_df['index'] = psi_df['index'].astype(str)
    psi_df.rename(columns = {'index':'ranges'}, inplace = True)
    # print(psi_df)
    ks_score = my_class.kstest('tot_cur_bal')

    if ks_score.pvalue <= 0.05:
        ks_string = str(ks_score.pvalue) + '. ' + ' Null hypothese (Training and Testing Set are from the same statistical distribution) is rejected.'
    else:
        ks_string = str(ks_score.pvalue) + '. ' + ' Null hypothese (Training and Testing Set are from the same statistical distribution) cannot be rejected.'
        
    return html.Div(
        style={'backgroundColor':'#fee6c8', 'width': '95%', 'margin': 'auto'}, children = [
        html.Div([html.H3('Probability Stability Index Table', style={'textAlign': 'center', 'fontWeight': 'bold'}),
                  html.H5(id = 'psi-score', 
                          children = 'PSI Score: '+str(psi_score), 
                          style={'textAlign': 'center', 'fontWeight': 'bold'})], 
                  style={'width': '100%', 'top': 0, 'left': 0, 'margin': 0}),
        html.Label('Please choose the number of bins to be sliced: '),
        dcc.Input(id="psi-num-of-bins",
                  type='number',
                  value=10,
                  min=2,
                  max=50,
                  # restrict into integer
                  step=1, 
                  pattern=r'\d*'),
        html.Br(),
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


def feature_metrics_layout(train_data: pd.DataFrame,
                           test_data: pd.DataFrame, 
                           ft_name_list: list[str], 
                           number_of_bins: int)->html.Div:
    
    stats_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(train_data, 
                                                                            test_data)
    
    csi_df, csi_dict = stats_class.csi_for_all_features(ft_name_list, number_of_bins)

    csi_children = []

    for df, ft_name in zip(csi_df, ft_name_list):
        df.columns = df.columns.astype(str)
        df = df.reset_index()
        df['index'] = df['index'].astype(str)
        df.rename(columns = {'index':'ranges'}, inplace = True)
        csi_children.append(html.Br())
        csi_children.append(html.Br())
        csi_children.append(html.H5(ft_name + ' CSI Score:    ' + str(csi_dict[ft_name]),
                                    style={'textAlign': 'center', 'fontWeight': 'bold'}))
        csi_children.append(dash_table.DataTable(id=ft_name+"-csi-table",
                                                 data= df.to_dict('records'), 
                                                 columns=[{"name": i, "id": i} for i in df.columns], 
                                                 sort_action='native'))
        
    
    return html.Div( 
        style={'backgroundColor': '#d7d7d7', 'width': '95%', 'margin': 'auto'}, children=[
        html.Br(),
        html.H3('Feature Metrics', style={'textAlign': 'left', 'fontWeight': 'bold'}),
        html.Label('Please choose the feature to be displayed: '),
        dcc.Dropdown(id="csi-feature-multi-dynamic-dropdown", 
                     multi=True,
                     options= dict(zip(train_data['raw_X'].columns.to_list(), 
                                       train_data['raw_X'].columns.to_list())),
                     value= [],
                     style={'width': '50%', 'margin': 'left'}),
        html.Label('Please choose the number of bins to be sliced: '),
        dcc.Input(id="csi-num-of-bins",
                  type='number',
                  value=10,
                  min=2,
                  max=50,
                  # restrict into integer
                  step=1, 
                  pattern=r'\d*'),
        html.Br(),
        html.Br(),
        html.H4('GINI Index: ' + str(0.71), style={'textAlign': 'left', 'fontWeight': 'bold'}),
        html.Div([
                    html.H4('Characteristic Stability Index Table', style={'textAlign': 'center', 'fontWeight': 'bold'})
        ], style={'width': '100%', 'top': 0, 'left': 0, 'margin': 0}),
        html.Div(id="feature_related_viz", children = csi_children),
        html.Br(),
        html.Br()        
    ]) 




