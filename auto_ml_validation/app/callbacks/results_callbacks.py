from auto_ml_validation.app.index import app
from auto_ml_validation.app.pages.results import *
from dash.dependencies import Input, Output, State
import json
# Body Content

# callbacks
@app.callback(
    [Output('dist-curve1', 'figure'), 
     Output('roc-curve1', 'figure'),
     Output('pr-curve1', 'figure'), 
     Output('metrics1', 'children')
     ],
    [Input('threshold1', 'value')]
)
def generate_bm_performance_metrics(threshold):
    with open('././models/lr.pkl', 'rb') as f:
        model = pickle.load(f)
    pme = performance_metrics_evaluator.PerformanceEvaluator(bm_test_data['pred_proba'],
                                                             float(threshold),
                                                             bm_test_data['y'],
                                                             bm_test_data['processed_X'],
                                                             model,)

    dist = pme.get_dist_plot()
    roc = pme.get_roc_curve()
    pr = pme.get_pr_curve()
    # lift = pme.get_lift_chart()
    metrics = pme.cal_metrics()
    # confusion_matrix = pme.get_confusion_matrix()

    metrics_comp = html.Div([
        html.H6(f'Accuracy {metrics["accuracy"]}'),
        html.H6(f'Precision {metrics["precision"]}'),
        html.H6(f'Recall {metrics["recall"]}'),
        html.H6(f'F1-Score {metrics["f1_score"]}'),
        ])

    return dist, roc, pr, metrics_comp

@app.callback(
    [Output('dist-curve2', 'figure'), 
     Output('roc-curve2', 'figure'),
     Output('pr-curve2', 'figure'), 
     Output('metrics2', 'children')
     ],
    [Input('threshold2', 'value')]
)
def generate_re_performance_metrics(threshold):
    with open('././models/lr.pkl', 'rb') as f:
        model = pickle.load(f)
    pme = performance_metrics_evaluator.PerformanceEvaluator(re_test_data['pred_proba'],
                                                             float(threshold),
                                                             re_test_data['y'],
                                                             re_test_data['processed_X'],
                                                             model)

    dist = pme.get_dist_plot()
    roc = pme.get_roc_curve()
    pr = pme.get_pr_curve()
    # lift = pme.get_lift_chart()
    metrics = pme.cal_metrics()
    # confusion_matrix = pme.get_confusion_matrix()

    metrics_comp = html.Div([
        html.H6(f'Accuracy {metrics["accuracy"]}'),
        html.H6(f'Precision {metrics["precision"]}'),
        html.H6(f'Recall {metrics["recall"]}'),
        html.H6(f'F1-Score {metrics["f1_score"]}'),
        ])

    return dist, roc, pr, metrics_comp


# Update the PSI table when user change num of bins
@app.callback(
    Output("psi-table1", "data"),
    Input("psi-num-of-bins1", "value")
)
# def update_statistical_metrics(num_of_bins):
def update_bm_psi_table(num_of_bins):
        my_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(bm_train_data, 
                                                                         bm_test_data)
        psi_score, psi_df = my_class.calculate_psi(num_of_bins)
        # reset index as columns for display
        psi_df.columns = psi_df.columns.astype(str)
        psi_df = psi_df.reset_index()
        psi_df['index'] = psi_df['index'].astype(str)
        psi_df.rename(columns = {'index':'ranges'}, inplace = True)
        psi_df = psi_df.round(2)

        return psi_df.to_dict('records')

# Update the PSI table when user change num of bins
@app.callback(
    Output("psi-table2", "data"),
    Input("psi-num-of-bins2", "value")
)
# def update_statistical_metrics(num_of_bins):
def update_re_psi_table(num_of_bins):
        my_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(re_train_data, 
                                                                         re_test_data)
        psi_score, psi_df = my_class.calculate_psi(num_of_bins)
        # reset index as columns for display
        psi_df.columns = psi_df.columns.astype(str)
        psi_df = psi_df.reset_index()
        psi_df['index'] = psi_df['index'].astype(str)
        psi_df.rename(columns = {'index':'ranges'}, inplace = True)
        psi_df = psi_df.round(2)

        return psi_df.to_dict('records')
        
# Update the PSI score when user change num of bins
@app.callback(
    Output("psi-score1", "children"),
    Input("psi-num-of-bins1", "value")
)
def update_bm_psi_score(num_of_bins):
        my_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(bm_train_data, bm_test_data)
        psi_score, psi_df = my_class.calculate_psi(num_of_bins)

        return 'PSI Score: '+ str(psi_score)

# Update the PSI score when user change num of bins
@app.callback(
    Output("psi-score2", "children"),
    Input("psi-num-of-bins2", "value")
)
def update_re_psi_score(num_of_bins):
        my_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(re_train_data, re_test_data)
        psi_score, psi_df = my_class.calculate_psi(num_of_bins)

        return 'PSI Score: '+ str(psi_score)

@app.callback(
    Output("gini_viz1", "children"),
    Input("gini-feature-multi-dynamic-dropdown1", "value"), 
)            
def update_bm_gini(ft_name_list: list[str]):
    stats_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(bm_train_data, bm_test_data)
    gini = stats_class.cal_feature_gini()
    
    gini_children = []
    
    for ft_name in ft_name_list:
              gini_children.append(html.H5(ft_name + ' GINI Index: ' + str(gini[ft_name])))

    return html.Div(id="gini-viz1", children = gini_children)

@app.callback(
    Output("gini_viz2", "children"),
    Input("gini-feature-multi-dynamic-dropdown2", "value"), 
)            
def update_re_gini(ft_name_list: list[str]):
    stats_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(re_train_data, re_test_data)
    gini = stats_class.cal_feature_gini()
    
    gini_children = []
    
    for ft_name in ft_name_list:
              gini_children.append(html.H5(ft_name + ' GINI Index: ' + str(gini[ft_name])))

    return html.Div(id="gini-viz2", children = gini_children)

@app.callback(
    Output("feature_related_viz1", "children"),
    Input("csi-feature-multi-dynamic-dropdown1", "value"),
    Input("csi-num-of-bins1", "value")
)
def update_bm_csi_metrics(feature_list, num_of_bins):
        
        stats_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(bm_train_data, 
                                                                                bm_test_data)
        
        csi_df, csi_dict = stats_class.csi_for_all_features(feature_list, num_of_bins)

        csi_children = []
        
        for df, ft_name in zip(csi_df, feature_list):
                df.columns = df.columns.astype(str)
                df = df.reset_index()
                df['index'] = df['index'].astype(str)
                df = df.round(2)
                df.rename(columns = {'index':'ranges'}, inplace = True)
                csi_children.append(html.Br())
                csi_children.append(html.Br())
                csi_children.append(html.H5(ft_name + ' CSI Score:    ' + str(csi_dict[ft_name]),
                                            style={'textAlign': 'center', 'fontWeight': 'bold'}))
                csi_children.append(dash_table.DataTable(id=ft_name+"-csi-table",
                                                        data= df.round(2).to_dict('records'), 
                                                        columns=[{"name": i, "id": i} for i in df.columns], 
                                                        sort_action='native'))
        return html.Div(id="feature_related_viz1", children = csi_children)

# Update the CSI when user select features
# options = train_data['raw_X'].columns.to_list()
@app.callback(
    Output("feature_related_viz2", "children"),
    Input("csi-feature-multi-dynamic-dropdown2", "value"),
    Input("csi-num-of-bins2", "value")
)
def update_re_csi_metrics(feature_list, num_of_bins):
        
        stats_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(re_train_data, 
                                                                                re_test_data)
        
        csi_df, csi_dict = stats_class.csi_for_all_features(feature_list, num_of_bins)

        csi_children = []
        
        for df, ft_name in zip(csi_df, feature_list):
                df.columns = df.columns.astype(str)
                df = df.reset_index()
                df['index'] = df['index'].astype(str)
                df.rename(columns = {'index':'ranges'}, inplace = True)
                df = df.round(2)
                csi_children.append(html.Br())
                csi_children.append(html.Br())
                csi_children.append(html.H5(ft_name + ' CSI Score:    ' + str(csi_dict[ft_name]),
                                            style={'textAlign': 'center', 'fontWeight': 'bold'}))
                csi_children.append(dash_table.DataTable(id=ft_name+"-csi-table",
                                                        data= df.to_dict('records'), 
                                                        columns=[{"name": i, "id": i} for i in df.columns], 
                                                        sort_action='native'))
        return html.Div(id="feature_related_viz2", children = csi_children)


# Layout
results_layout = html.Div(children=[
    html.H1('Result for Replicated Model', style={'text-align': 'center'}),
    html.Br(),
    re_performance_metric_layout(),
    # performance_metric_layout(test_data, '././models/lr.pkl'),
    html.Br(),
    re_statistical_model_metrics_layout(re_train_data, re_test_data, 10),
    html.Br(),
    html.Br(),
    re_gini_layout(re_train_data, re_test_data,[]),
    re_csi_table_layout(re_train_data),
    html.Br(),
    html.Br(),
    # re_trans_layout(re_train_data, '././models/lr.pkl'),   
    html.Br(),
    html.H1('Result for Benchmark Model', style={'text-align': 'center'}), 
    html.Br(),
    bm_performance_metric_layout(),
    html.Br(),
    bm_statistical_model_metrics_layout(bm_train_data, bm_test_data, 10),
    html.Br(),
    html.Br(),
    bm_gini_layout(bm_train_data, bm_test_data,[]),
    bm_csi_table_layout(bm_train_data),
    html.Br(),
    html.Br(),
    bm_trans_layout(bm_train_data, '././models/lr.pkl'),  
    html.Br(),
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

