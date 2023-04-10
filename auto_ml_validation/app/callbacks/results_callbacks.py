import matplotlib.pyplot as plt
from auto_ml_validation.app.index import app
from auto_ml_validation.app.pages.results import *
from dash.dependencies import Input, Output, State
import json
from ...validation_package.evaluation import performance_metrics_evaluator as pme
from ...validation_package.evaluation_pipeline import evaluation_pipeline
from ...validation_package.report.generate_report import generate_report

plt.switch_backend('Agg')

# Layout
re_header, bm_header = sticky_headers()

re_layout = html.Div(
    children=[
        re_header,
        html.Br(),
        re_performance_metric_layout(),
        re_statistical_model_metrics_layout(),
        html.Br(),
        re_gini_layout(),
        re_csi_table_layout(),
        html.Br(),
        re_trans_layout(),
    ],
    style={'width': '100%', 'display': 'inline-block',
           'vertical-align': 'top', },
)

bm_layout = html.Div(
    children=[
        bm_header,
        html.Br(),
        bm_performance_metric_layout(),
        bm_statistical_model_metrics_layout(),
        html.Br(),
        bm_gini_layout(),
        bm_csi_table_layout(),
        html.Br(),
        bm_trans_layout(),
    ],
    style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top'},
)

results_layout = html.Div(
    id="results-layout",
    children=[
        html.Div(download_report_layout(), style={
                 'float': 'right', "margin-right": "30px"}),
        html.Br(),
        html.Div([re_layout, bm_layout], style={
                 'display': 'flex',  'clear': 'right'}),
        dcc.Store(id='re_dist_path_st', data=None, storage_type='session'),
        dcc.Store(id='re_roc_path_st', data=None, storage_type='session'),
        dcc.Store(id='re_pr_path_st', data=None, storage_type='session'),
        dcc.Store(id='re_metrics_st', data=None, storage_type='session'),
        dcc.Store(id='re_auc_st', data=None, storage_type='session'),
        dcc.Store(id='re_psi_score_st', data=None, storage_type='session'),
        dcc.Store(id='re_psi_df_st', data=None, storage_type='session'),
        dcc.Store(id='re_ks_st', data=None, storage_type='session'),
        dcc.Store(id='re_csi_dfs_st', data=None, storage_type='session'),
        dcc.Store(id='re_csi_dicts_st', data=None, storage_type='session'),
        dcc.Store(id='re_ft_gini_st', data=None, storage_type='session'),
        dcc.Store(id='re_lime_path_st', data=None, storage_type='session'),
        dcc.Store(id='re_shap_path_st', data=None, storage_type='session'),
        dcc.Store(id='re_lime_lst_st', data=None, storage_type='session'),
        dcc.Store(id='re_shap_lst_st', data=None, storage_type='session'),
    ]
)
# Callbacks
# Output generic performance metrics for both Model Replication and Auto-Benchmark


@app.callback(
    Output('dist-curve', 'figure'),
    Output('roc-curve', 'figure'),
    Output('rocauc-bm', 'children'),
    Output('pr-curve', 'figure'),
    Output('prauc-bm', 'children'),
    Output('metrics', 'children'),
    Input('threshold', 'value'),
    State('validator-input-trigger', "data"),
    State('validator-input-file', 'data'),
    State("validator-bm-model", "data"),
)
def bm_performance_metrics(threshold_bm, trigger, input_path, bm_path):
    if trigger:
        print("Generating benchmark performance metrics.")
        try:
            with open(f'././{input_path}', 'rb') as f:
                data = pickle.load(f)
            # Benchmark Output
            bm_test_data = data['bm_other_data']['Test']

            with open(f'././{bm_path}', 'rb') as f:
                bm_model = pickle.load(f)

            pme_obj = pme.PerformanceEvaluator(bm_test_data['pred_proba'],
                                               float(threshold_bm),
                                               bm_test_data['y'],
                                               bm_test_data['processed_X'],
                                               bm_model)

            dist_bm = pme_obj.get_dist_plot()
            roc_bm = pme_obj.get_roc_curve()
            pr_bm = pme_obj.get_pr_curve()
            metrics = pme_obj.cal_metrics()
            auc = pme_obj.cal_auc()
            rocauc = f"ROC-AUC: {round(auc['ROCAUC'], 3)}"
            prauc = f"PR-AUC: {round(auc['PRAUC'], 3)}"

            metrics_comp = html.Div([
                html.H6(f'Accuracy {metrics["accuracy"]}'),
                html.H6(f'Precision {metrics["precision"]}'),
                html.H6(f'Recall {metrics["recall"]}'),
                html.H6(f'F1-Score {metrics["f1_score"]}'),
            ])
            return dist_bm, roc_bm, rocauc, pr_bm, prauc, metrics_comp
        except Exception as e:
            print(e)
            return (None,)*6
    return (None,)*6


@app.callback(
    Output('dist-curve-re', 'figure'),
    Output('roc-curve-re', 'figure'),
    Output('rocauc-re', 'children'),
    Output('pr-curve-re', 'figure'),
    Output('prauc-re', 'children'),
    Output('metrics-re', 'children'),
    Output('re_metrics_st', 'data'),
    Output('re_dist_path_st', 'data'),
    Output('re_roc_path_st', 'data'),
    Output('re_pr_path_st', 'data'),
    Output('re_auc_st', 'data'),
    Input('threshold-re', 'value'),
    Input('validator-input-trigger', 'data'),
    Input('validator-input-file', 'data'),
    Input("validator-rep-model", "data"),
)
def re_performance_metrics(threshold_re, trigger, input_path, rep_path):
    if trigger:
        print("Generating replication performance metrics.")
        try:
            with open(f'././{input_path}', 'rb') as f:
                data = pickle.load(f)
            # Replication Output
            re_test_data = data['re_other_data']['Test']

            with open(f'././{rep_path}', 'rb') as f:
                re_model = pickle.load(f)

            pme_obj = pme.PerformanceEvaluator(re_test_data['pred_proba'],
                                               float(threshold_re),
                                               re_test_data['y'],
                                               re_test_data['processed_X'],
                                               re_model)

            dist_re = pme_obj.get_dist_plot()
            dist_path = '././auto_ml_validation/app/assets/images/dist_re.png'
            dist_re.write_image(dist_path)
            roc_re = pme_obj.get_roc_curve()
            roc_path = '././auto_ml_validation/app/assets/images/roc_re.png'
            roc_re.write_image(roc_path)
            pr_re = pme_obj.get_pr_curve()
            pr_path = '././auto_ml_validation/app/assets/images/pr_re.png'
            pr_re.write_image(pr_path)
            metrics_re = pme_obj.cal_metrics()
            auc = pme_obj.cal_auc()
            rocauc = f"ROC-AUC: {round(auc['ROCAUC'], 3)}"
            prauc = f"PR-AUC: {round(auc['PRAUC'], 3)}"

            metrics_comp_re = html.Div([
                html.H6(f'Accuracy {metrics_re["accuracy"]}'),
                html.H6(f'Precision {metrics_re["precision"]}'),
                html.H6(f'Recall {metrics_re["recall"]}'),
                html.H6(f'F1-Score {metrics_re["f1_score"]}'),
            ])
            return dist_re, roc_re, rocauc, pr_re, prauc, metrics_comp_re, metrics_re, dist_path, roc_path, pr_path, auc
        except Exception as e:
            print(e)
            return (None,)*11
    return (None,)*11

# Update benchmark threshold value


@app.callback(
    Output('threshold-text', 'children'),
    Input('threshold', 'value')
)
def update_threshold_text_bm(value):
    return 'Adjust the threshold here: %.2f' % float(value)

# Update benchmark threshold value


@app.callback(
    Output('threshold-text-re', 'children'),
    Input('threshold-re', 'value')
)
def update_threshold_text_re(value):
    return 'Adjust the threshold here: %.2f' % float(value)

# Output PSI and KS


@app.callback(
    Output("psi-table", "data"),
    Output("psi-score", "children"),
    Output('psi-table', 'columns'),
    Output("ks-tests", "children"),
    Input("psi-num-of-bins", "value"),
    Input('validator-input-trigger', 'data'),
    Input('validator-input-file', 'data'),
)
def bm_psi_ks(num_of_bins, trigger, input_path):
    if trigger:
        print("Generating benchmark PSI.")
        try:
            with open(f'././{input_path}', 'rb') as f:
                data = pickle.load(f)
            bm_train_data = data['bm_train_data']
            bm_test_data = data['bm_other_data']['Test']

            # Auto-Benchmarking
            my_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(bm_train_data,
                                                                                 bm_test_data)
            psi_score, psi_df = my_class.calculate_psi(num_of_bins)
            psi_score_text = 'PSI Score: ' + str(round(psi_score, 3))
            psi_df.columns = psi_df.columns.astype(str)
            psi_df = psi_df.reset_index()
            psi_df['index'] = psi_df['index'].astype(str)
            psi_df.rename(columns={'index': 'ranges'}, inplace=True)

            ks_dict = my_class.kstest()
            ks_output = [html.H3('Kolmogorov–Smirnov statistic'),
                         html.H6(
                'Quantifies a distance of the distribution within the training sample and testing sample, or between the two.'),
                html.H6('KS Train: ' + str(ks_dict['Train']), style={
                    'textAlign': 'left', 'fontWeight': 'bold'}),
                html.H6('KS Test: ' + str(ks_dict['Test']), style={
                    'textAlign': 'left', 'fontWeight': 'bold'}),
                html.H6('KS Train & Test: ' + str(ks_dict['Train vs Test']), style={'textAlign': 'left', 'fontWeight': 'bold'}),]

            return psi_df.to_dict('records'), psi_score_text, [{"name": col, "id": col} for col in psi_df.columns], ks_output
        except Exception as e:
            print(e)
            return None, None, None, None
    return None, None, None, None


@app.callback(
    Output("psi-table-re", "data"),
    Output("psi-score-re", "children"),
    Output('psi-table-re', 'columns'),
    Output("ks-tests-re", "children"),
    Output('re_psi_score_st', 'data'),
    Output('re_psi_df_st', 'data'),
    Output('re_ks_st', 'data'),
    Input("psi-num-of-bins-re", "value"),
    Input('validator-input-trigger', 'data'),
    Input('validator-input-file', 'data'),
)
def re_psi_ks(num_of_bins, trigger, input_path):
    if trigger:
        print("Generating replication PSI.")
        try:
            with open(f'././{input_path}', 'rb') as f:
                data = pickle.load(f)
            re_train_data = data['re_train_data']
            re_test_data = data['re_other_data']['Test']

            # Model Replication
            my_class_re = statistical_metrics_evaluator.StatisticalMetricsEvaluator(re_train_data,
                                                                                    re_test_data)
            psi_score_re, psi_df_re = my_class_re.calculate_psi(num_of_bins)
            re_psi_df_st = psi_df_re.to_json(orient='split')
            psi_score_text_re = 'PSI Score: ' + str(round(psi_score_re, 3))
            psi_df_re.columns = psi_df_re.columns.astype(str)
            psi_df_re = psi_df_re.reset_index()
            psi_df_re['index'] = psi_df_re['index'].astype(str)
            psi_df_re.rename(columns={'index': 'ranges'}, inplace=True)

            ks_dict_re = my_class_re.kstest()
            ks_output_re = [html.H3('Kolmogorov–Smirnov statistic'),
                            html.H6(
                                'Quantifies a distance of the distribution within the training sample and testing sample, or between the two.'),
                            html.H6('KS Train: ' + str(ks_dict_re['Train']), style={
                                    'textAlign': 'left', 'fontWeight': 'bold'}),
                            html.H6('KS Test: ' + str(ks_dict_re['Test']), style={
                                    'textAlign': 'left', 'fontWeight': 'bold'}),
                            html.H6('KS Train & Test: ' + str(ks_dict_re['Train vs Test']), style={'textAlign': 'left', 'fontWeight': 'bold'}),]
            return psi_df_re.to_dict('records'), psi_score_text_re, [{"name": col, "id": col} for col in psi_df_re.columns], ks_output_re, psi_score_re, re_psi_df_st, ks_dict_re
        except Exception as e:
            print(e)
            return (None,) * 7
    return (None,) * 7
# Update gini features selection based on train dataset for both models


@app.callback(
    Output("gini-feature-multi-dynamic-dropdown",
           "options"), Output("gini-feature-multi-dynamic-dropdown-re", "options"),
    Input('validator-input-trigger', 'data'),
    Input('validator-input-file', 'data'),
)
def update_gini(trigger, input_path):
    if trigger:
        with open(f'././{input_path}', 'rb') as f:
            data = pickle.load(f)
        bm_train_data = data['bm_train_data']
        re_train_data = data['re_train_data']
        return dict(zip(bm_train_data['raw_X'].columns.to_list(), bm_train_data['raw_X'].columns.to_list())), dict(zip(re_train_data['raw_X'].columns.to_list(), re_train_data['raw_X'].columns.to_list()))
    return []

# Output gini metric for benchmark model


@app.callback(
    Output("gini-viz", "children"),
    Input("gini-feature-multi-dynamic-dropdown", "value"),
    Input('validator-input-trigger', 'data'),
    Input('validator-input-file', 'data'),
)
def update_bm_gini(ft_name_list: list[str], trigger, input_path):
    if trigger:
        with open(f'././{input_path}', 'rb') as f:
            data = pickle.load(f)
        bm_train_data = data['bm_train_data']
        bm_test_data = data['bm_other_data']['Test']
        stats_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(
            bm_train_data, bm_test_data)
        gini = stats_class.cal_feature_gini()

        gini_children = []

        for ft_name in ft_name_list:
            gini_children.append(
                html.H5(ft_name + ' GINI Index: ' + str(gini[ft_name])))

        return gini_children

# Output gini metric for model replication


@app.callback(
    Output("gini-viz-re", "children"),
    Output('re_ft_gini_st', 'data'),
    Input("gini-feature-multi-dynamic-dropdown-re", "value"),
    Input('validator-input-trigger', 'data'),
    Input('validator-input-file', 'data'),
)
def update_re_gini(ft_name_list: list[str], trigger, input_path):
    if trigger:
        with open(f'././{input_path}', 'rb') as f:
            data = pickle.load(f)
        re_train_data = data['re_train_data']
        re_test_data = data['re_other_data']['Test']
        stats_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(
            re_train_data, re_test_data)
        gini = stats_class.cal_feature_gini()

        gini_children = []

        for ft_name in ft_name_list:
            gini_children.append(
                html.H5(ft_name + ' GINI Index: ' + str(gini[ft_name])))

        return gini_children, gini

# Generate and populate csi feature metrics for both models


@app.callback(
    Output("csi-feature-multi-dynamic-dropdown",
           "options"), Output("csi-feature-multi-dynamic-dropdown-re", "options"),
    Input('validator-input-trigger', 'data'),
    Input('validator-input-file', 'data')
)
def update_gini_bm(trigger, input_path):
    if trigger:
        with open(f'././{input_path}', 'rb') as f:
            data = pickle.load(f)
        bm_train_data = data['bm_train_data']
        re_train_data = data['re_train_data']
        return dict(zip(bm_train_data['raw_X'].columns.to_list(), bm_train_data['raw_X'].columns.to_list())), dict(zip(re_train_data['raw_X'].columns.to_list(), re_train_data['raw_X'].columns.to_list()))
    return []


# Update the CSI when user select features
@app.callback(
    Output("feature-related-viz", "children"),
    Input("csi-feature-multi-dynamic-dropdown", "value"),
    Input("csi-num-of-bins", "value"),
    Input('validator-input-trigger', 'data'),
    Input('validator-input-file', 'data'),
)
def update_csi_metrics_bm(feature_list, num_of_bins, trigger, input_path):
    if trigger:
        with open(f'././{input_path}', 'rb') as f:
            data = pickle.load(f)
        bm_train_data = data['bm_train_data']
        bm_test_data = data['bm_other_data']['Test']
        stats_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(bm_train_data,
                                                                                bm_test_data)

        csi_df, csi_dict = stats_class.csi_for_all_features(
            feature_list, num_of_bins)

        csi_children = []

        for df, ft_name in zip(csi_df, feature_list):
            df.columns = df.columns.astype(str)
            df = df.reset_index()
            df['index'] = df['index'].astype(str)
            df.rename(columns={'index': 'ranges'}, inplace=True)
            csi_children.append(html.Br())
            csi_children.append(html.Br())
            csi_children.append(html.H5(ft_name + ' CSI Score:    ' + str(csi_dict[ft_name]),
                                        style={'textAlign': 'center', 'fontWeight': 'bold'}))
            csi_children.append(dash_table.DataTable(id=ft_name+"-csi-table",
                                                     data=df.to_dict(
                                                         'records'),
                                                     columns=[{"name": i, "id": i}
                                                              for i in df.columns],
                                                     sort_action='native'))
        return csi_children


# Update the CSI when user select features


@app.callback(
    Output("feature-related-viz-re", "children"),
    Output('re_csi_dfs_st', 'data'),
    Output('re_csi_dicts_st', 'data'),
    Input("csi-feature-multi-dynamic-dropdown-re", "value"),
    Input("csi-num-of-bins-re", "value"),
    Input('validator-input-trigger', 'data'),
    Input('validator-input-file', 'data'),
)
def update_csi_metrics_re(feature_list, num_of_bins, trigger, input_path):
    if trigger:
        with open(f'././{input_path}', 'rb') as f:
            data = pickle.load(f)
        re_train_data = data['re_train_data']
        re_test_data = data['re_other_data']['Test']
        stats_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(re_train_data,
                                                                                re_test_data)

        csi_df, csi_dict = stats_class.csi_for_all_features(
            feature_list, num_of_bins)

        csi_children, csi_json_l = [], []

        for df, ft_name in zip(csi_df, feature_list):
            csi_json_l.append(df.to_json(orient='split'))
            df.columns = df.columns.astype(str)
            df = df.reset_index()
            df['index'] = df['index'].astype(str)
            df.rename(columns={'index': 'ranges'}, inplace=True)
            csi_children.append(html.Br())
            csi_children.append(html.Br())
            csi_children.append(html.H5(ft_name + ' CSI Score:    ' + str(csi_dict[ft_name]),
                                        style={'textAlign': 'center', 'fontWeight': 'bold'}))
            csi_children.append(dash_table.DataTable(id=ft_name+"-csi-table",
                                                     data=df.to_dict(
                                                         'records'),
                                                     columns=[{"name": i, "id": i}
                                                              for i in df.columns],
                                                     sort_action='native'))
        return csi_children, csi_json_l, csi_dict

# Output transparency metrics


@app.callback(
    Output("global-lime", "src"),
    Output("global-shap", "src"),
    Input('validator-input-trigger', 'data'),
    State('validator-input-file', 'data'),
    State("validator-bm-model", "data"),
)
def bm_transparency_plots(trigger, input_path, bm_path):
    if trigger:
        print("Generate benchmark transparency plots.")
        with open(f'././{input_path}', 'rb') as f:
            data = pickle.load(f)
        with open(f'././{bm_path}', 'rb') as f:
            bm_model = pickle.load(f)

        bm_train_data = data['bm_train_data']
        try:
            evaluator = transparency_metrics_evaluator.TransparencyMetricsEvaluator(
                bm_model, bm_train_data['processed_X'].sample(100))  # Too large, hence we take a sample
            local_lime_fig, global_lime_fig, local_text_lime, global_text_lime = evaluator.lime_interpretability()
            global_lime_fig.savefig(
                '././auto_ml_validation/app/assets/images/global_lime_bm.png', bbox_inches='tight')

            local_shap_fig, global_shap_fig, local_text_shap, global_text_shap = evaluator.shap_interpretability()

            global_shap_fig.savefig(
                '././auto_ml_validation/app/assets/images/global_shap_bm.png', bbox_inches='tight')
            global_lime_bm = app.get_asset_url("images/global_lime_bm.png")
            global_shap_bm = app.get_asset_url("images/global_shap_bm.png")
            return global_lime_bm, global_shap_bm
        except Exception as e:
            print(e)
            return "", ""
    return "", ""


@app.callback(
    Output("global-lime-re", "src"),
    Output("global-shap-re", "src"),
    Output('re_lime_path_st', 'data'),
    Output('re_shap_path_st', 'data'),
    Output('re_lime_lst_st', 'data'),
    Output('re_shap_lst_st', 'data'),
    Input('validator-input-trigger', 'data'),
    Input('validator-input-file', 'data'),
    Input("validator-rep-model", "data"),
)
def re_transparency_plots(trigger, input_path, rep_path):
    if trigger:
        print("Generate replication transparency plots.")
        with open(f'././{input_path}', 'rb') as f:
            data = pickle.load(f)
        with open(f'././{rep_path}', 'rb') as f:
            re_model = pickle.load(f)

        re_train_data = data['re_train_data']
        try:
            evaluator = transparency_metrics_evaluator.TransparencyMetricsEvaluator(
                re_model, re_train_data['processed_X'].sample(100))

            local_lime_fig, global_lime_fig, local_text_lime, global_lime_lst = evaluator.lime_interpretability()
            global_lime_path = '././auto_ml_validation/app/assets/images/global_lime_re.png'
            global_lime_fig.savefig(global_lime_path, bbox_inches='tight')

            local_shap_fig, global_shap_fig, local_text_shap, global_shap_lst = evaluator.shap_interpretability()
            global_shap_path = '././auto_ml_validation/app/assets/images/global_shap_re.png'
            global_shap_fig.savefig(global_shap_path, bbox_inches='tight')
            global_lime_re = app.get_asset_url("images/global_lime_re.png")

            global_shap_re = app.get_asset_url("images/global_shap_re.png")
        except Exception as e:
            print(e)
            return "", ""
        return global_lime_re, global_shap_re, global_lime_path, global_shap_path, global_lime_lst, global_shap_lst

# Run the evaluation pipeline and generate word doc report


@app.callback(
    Output('report-message', 'children'),
    Input('download-report', 'n_clicks'),
    State('re_metrics_st', 'data'),
    State('re_dist_path_st', 'data'),
    State('re_roc_path_st', 'data'),
    State('re_pr_path_st', 'data'),
    State('re_auc_st', 'data'),
    State('re_psi_score_st', 'data'),
    State('re_psi_df_st', 'data'),
    State('re_ks_st', 'data'),
    State('re_csi_dfs_st', 'data'),
    State('re_csi_dicts_st', 'data'),
    State('re_ft_gini_st', 'data'),
    State('re_lime_path_st', 'data'),
    State('re_shap_path_st', 'data'),
    State('re_lime_lst_st', 'data'),
    State('re_shap_lst_st', 'data'),
    prevent_initial_call=True,
)
def convert_to_report(
    n_clicks,
    re_metrics,
    re_dist_path,
    re_roc_path,
    re_pr_path,
    re_auc,
    re_psi_score,
    re_psi_df,
    re_ks,
    re_csi_list,
    re_csi_dict,
    re_ginis,
    re_lime_path,
    re_shap_path,
    re_lime_lst,
    re_shap_lst,
):
    if n_clicks:
        re_psi_df = pd.read_json(re_psi_df, orient='split')
        re_csi_list = [pd.read_json(df, orient='split') for df in re_csi_list]
        re_eval_outputs = {}
        re_eval_outputs['Test'] = {}
        re_eval_outputs['Test']['charts'] = {
            'dist': re_dist_path,
            'pr': re_pr_path,
            'roc': re_roc_path,
            'global_lime': re_lime_path,
            'global_shap': re_shap_path,
        }
        re_eval_outputs['Test']['txt'] = {
            'metrics': re_metrics,
            'feature_gini': re_ginis,
            'auc': re_auc,
            'global_lime': re_lime_lst,
            'global_shap': re_shap_lst,
            'psi': re_psi_score,
            'psi_df': re_psi_df,
            'csi_list': re_csi_list,
            'csi_dict': re_csi_dict,
            'ks': re_ks
        }

        generate_report(re_eval_outputs,
                        report_path='./outputs/test_bc2_report.docx')
        return "Report generated!"
