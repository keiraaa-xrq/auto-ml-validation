import json
import os
import matplotlib.pyplot as plt
from auto_ml_validation.app.index import app
from auto_ml_validation.app.pages.results import *
from dash.dependencies import Input, Output, State
from ...validation_package.evaluation.performance_metrics_evaluator import PerformanceEvaluator
from ...validation_package.evaluation.statistical_metrics_evaluator import StatisticalMetricsEvaluator
from ...validation_package.evaluation.transparency_metrics_evaluator import TransparencyMetricsEvaluator
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

        dcc.Store(id='bm_dist_path_st', data=None, storage_type='session'),
        dcc.Store(id='bm_roc_path_st', data=None, storage_type='session'),
        dcc.Store(id='bm_pr_path_st', data=None, storage_type='session'),
        dcc.Store(id='bm_metrics_st', data=None, storage_type='session'),
        dcc.Store(id='bm_auc_st', data=None, storage_type='session'),
        dcc.Store(id='bm_psi_score_st', data=None, storage_type='session'),
        dcc.Store(id='bm_psi_df_st', data=None, storage_type='session'),
        dcc.Store(id='bm_ks_st', data=None, storage_type='session'),
        dcc.Store(id='bm_ft_gini_st', data=None, storage_type='session'),
        dcc.Store(id='bm_lime_path_st', data=None, storage_type='session'),
        dcc.Store(id='bm_shap_path_st', data=None, storage_type='session'),
        dcc.Store(id='bm_lime_lst_st', data=None, storage_type='session'),
        dcc.Store(id='bm_shap_lst_st', data=None, storage_type='session'),
    ]
)
# Callbacks
# Output generic performance metrics for both Model Replication and Auto-Benchmark


def get_performance_metrics(threshold, input_path, model_path, prefix):
    try:
        with open(f'{input_path}', 'rb') as f:
            data = pickle.load(f)

        test_data = data[f'{prefix}_other_data']['Test']

        with open(f'{model_path}', 'rb') as f:
            model = pickle.load(f)

        pme_obj = PerformanceEvaluator(test_data['pred_proba'],
                                       float(threshold),
                                       test_data['y'],
                                       test_data['processed_X'],
                                       model)

        dist = pme_obj.get_dist_plot()
        dist_path = f'./auto_ml_validation/app/assets/images/dist_{prefix}.png'
        dist.write_image(dist_path)
        roc = pme_obj.get_roc_curve()
        roc_path = f'./auto_ml_validation/app/assets/images/roc_{prefix}.png'
        roc.write_image(roc_path)
        pr = pme_obj.get_pr_curve()
        pr_path = f'./auto_ml_validation/app/assets/images/pr_{prefix}.png'
        pr.write_image(pr_path)
        metrics = pme_obj.cal_metrics()
        auc = pme_obj.cal_auc()
        rocauc = f"ROC-AUC: {round(auc['ROCAUC'], 3)}"
        prauc = f"PR-AUC: {round(auc['PRAUC'], 3)}"

        metrics_comp_re = html.Div([
            html.H6(f'Accuracy {metrics["accuracy"]}'),
            html.H6(f'Precision {metrics["precision"]}'),
            html.H6(f'Recall {metrics["recall"]}'),
            html.H6(f'F1-Score {metrics["f1_score"]}'),
        ])
        return dist, roc, rocauc, pr, prauc, metrics_comp_re, metrics, dist_path, roc_path, pr_path, auc
    except Exception as e:
        print(e)
        return (None,)*11


@app.callback(
    Output('dist-curve', 'figure'),
    Output('roc-curve', 'figure'),
    Output('rocauc-bm', 'children'),
    Output('pr-curve', 'figure'),
    Output('prauc-bm', 'children'),
    Output('metrics', 'children'),
    Output('bm_metrics_st', 'data'),
    Output('bm_dist_path_st', 'data'),
    Output('bm_roc_path_st', 'data'),
    Output('bm_pr_path_st', 'data'),
    Output('bm_auc_st', 'data'),
    Input('threshold', 'value'),
    State('validator-input-trigger', "data"),
    State('validator-input-file', 'data'),
    State("validator-bm-model", "data"),
)
def bm_performance_metrics(threshold_bm, trigger, input_path, bm_path):
    if trigger:
        print("Generating benchmark performance metrics.")
        return get_performance_metrics(threshold_bm, input_path, bm_path, 'bm')
    return (None,) * 11


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
        return get_performance_metrics(threshold_re, input_path, rep_path, 're')
    return (None,) * 11

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


def get_psi_ks(num_of_bins, input_path, prefix):
    try:
        with open(f'{input_path}', 'rb') as f:
            data = pickle.load(f)
        train_data = data[f'{prefix}_train_data']
        test_data = data[f'{prefix}_other_data']['Test']

        stats_eval = StatisticalMetricsEvaluator(train_data, test_data)

        psi_score, psi_df = stats_eval.calculate_psi(num_of_bins)
        psi_df_st = psi_df.to_json(orient='split')
        psi_score_text = 'PSI Score: ' + str(round(psi_score, 3))
        psi_df.columns = psi_df.columns.astype(str)
        psi_df = psi_df.reset_index()
        psi_df['index'] = psi_df['index'].astype(str)
        psi_df.rename(columns={'index': 'ranges'}, inplace=True)

        ks_dict = stats_eval.kstest()
        ks_output = [
            html.H3('Kolmogorov–Smirnov statistic'),
            html.H6(
                'Quantifies a distance of the distribution within the training sample and testing sample, or between the two.'),
            html.H6('KS Train: ' + str(ks_dict['Train']), style={
                    'textAlign': 'left', 'fontWeight': 'bold'}),
            html.H6('KS Test: ' + str(ks_dict['Test']), style={
                    'textAlign': 'left', 'fontWeight': 'bold'}),
            html.H6('KS Train & Test: ' + str(ks_dict['Train vs Test']), style={
                    'textAlign': 'left', 'fontWeight': 'bold'}),
        ]
        return psi_df.to_dict('records'), psi_score_text, [{"name": col, "id": col} for col in psi_df.columns], ks_output, psi_score, psi_df_st, ks_dict
    except Exception as e:
        print(e)
        return (None,) * 7


@app.callback(
    Output("psi-table", "data"),
    Output("psi-score", "children"),
    Output('psi-table', 'columns'),
    Output("ks-tests", "children"),
    Output('bm_psi_score_st', 'data'),
    Output('bm_psi_df_st', 'data'),
    Output('bm_ks_st', 'data'),
    Input("psi-num-of-bins", "value"),
    Input('validator-input-trigger', 'data'),
    Input('validator-input-file', 'data'),
)
def bm_psi_ks(num_of_bins, trigger, input_path):
    if trigger:
        print("Generating benchmark PSI.")
        return get_psi_ks(num_of_bins, input_path, 'bm')
    return (None,) * 7


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
        return get_psi_ks(num_of_bins, input_path, 're')
    return (None,) * 7
# Update gini features selection based on train dataset for both models


@app.callback(
    Output("gini-feature-multi-dynamic-dropdown", "options"),
    Output("gini-feature-multi-dynamic-dropdown-re", "options"),
    Input('validator-input-trigger', 'data'),
    Input('validator-input-file', 'data'),
)
def update_gini_dropdown(trigger, input_path):
    if trigger:
        with open(f'{input_path}', 'rb') as f:
            data = pickle.load(f)
        bm_train_data = data['bm_train_data']
        re_train_data = data['re_train_data']
        return dict(zip(bm_train_data['raw_X'].columns.to_list(), bm_train_data['raw_X'].columns.to_list())), dict(zip(re_train_data['raw_X'].columns.to_list(), re_train_data['raw_X'].columns.to_list()))
    return []

# Output gini metric for benchmark model


def get_gini(ft_names, input_path, prefix):
    try:
        with open(f'{input_path}', 'rb') as f:
            data = pickle.load(f)
        train_data = data[f'{prefix}_train_data']
        test_data = data[f'{prefix}_other_data']['Test']
        stats_class = StatisticalMetricsEvaluator(train_data, test_data)
        gini = stats_class.cal_feature_gini()

        gini_children = []

        for ft_name in ft_names:
            gini_children.append(
                html.H5(ft_name + ' GINI Index: ' + str(gini[ft_name])))
        return gini_children, gini
    except Exception as e:
        print(e)
        return (None,) * 2


@app.callback(
    Output("gini-viz", "children"),
    Output('bm_ft_gini_st', 'data'),
    Input("gini-feature-multi-dynamic-dropdown", "value"),
    Input('validator-input-trigger', 'data'),
    Input('validator-input-file', 'data'),
)
def update_bm_gini(ft_name_list: list[str], trigger, input_path):
    if trigger:
        return get_gini(ft_name_list, input_path, 'bm')
    return (None,) * 2


@app.callback(
    Output("gini-viz-re", "children"),
    Output('re_ft_gini_st', 'data'),
    Input("gini-feature-multi-dynamic-dropdown-re", "value"),
    Input('validator-input-trigger', 'data'),
    Input('validator-input-file', 'data'),
)
def update_re_gini(ft_name_list: list[str], trigger, input_path):
    if trigger:
        return get_gini(ft_name_list, input_path, 're')
    return (None,) * 2
# Generate and populate csi feature metrics for both models


@app.callback(
    Output("csi-feature-multi-dynamic-dropdown", "options"),
    Output("csi-feature-multi-dynamic-dropdown-re", "options"),
    Input('validator-input-trigger', 'data'),
    Input('validator-input-file', 'data')
)
def update_csi_dropdown(trigger, input_path):
    if trigger:
        with open(f'{input_path}', 'rb') as f:
            data = pickle.load(f)
        bm_train_data = data['bm_train_data']
        re_train_data = data['re_train_data']
        return dict(zip(bm_train_data['raw_X'].columns.to_list(), bm_train_data['raw_X'].columns.to_list())), dict(zip(re_train_data['raw_X'].columns.to_list(), re_train_data['raw_X'].columns.to_list()))
    return []


def get_csi(features, num_of_bins, input_path, prefix):
    try:
        with open(f'{input_path}', 'rb') as f:
            data = pickle.load(f)
        train_data = data[f'{prefix}_train_data']
        test_data = data[f'{prefix}_other_data']['Test']
        stats_class = StatisticalMetricsEvaluator(train_data, test_data)

        csi_df, csi_dict = stats_class.csi_for_all_features(
            features, num_of_bins)

        csi_children, csi_json_l = [], []

        for df, ft_name in zip(csi_df, features):
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
    except Exception as e:
        print(e)
        return (None,) * 3

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
        return get_csi(feature_list, num_of_bins, input_path, 'bm')
    return (None,) * 3


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
        return get_csi(feature_list, num_of_bins, input_path, 're')
    return (None,) * 3


# Output transparency metrics
def get_transparency_plots(input_path, model_path, prefix):
    try:
        with open(f'{input_path}', 'rb') as f:
            data = pickle.load(f)
        with open(f'{model_path}', 'rb') as f:
            model = pickle.load(f)

        train_data = data[f'{prefix}_train_data']

        evaluator = TransparencyMetricsEvaluator(
            model, train_data['processed_X'].sample(100))

        local_lime_fig, global_lime_fig, local_text_lime, global_lime_lst = evaluator.lime_interpretability()
        global_lime_path = f'./auto_ml_validation/app/assets/images/global_lime_{prefix}.png'
        global_lime_fig.savefig(global_lime_path, bbox_inches='tight')

        local_shap_fig, global_shap_fig, local_text_shap, global_shap_lst = evaluator.shap_interpretability()
        global_shap_path = f'./auto_ml_validation/app/assets/images/global_shap_{prefix}.png'
        global_shap_fig.savefig(global_shap_path, bbox_inches='tight')
        global_lime_url = app.get_asset_url(f"images/global_lime_{prefix}.png")
        global_shap_url = app.get_asset_url(f"images/global_shap_{prefix}.png")
        return global_lime_url, global_shap_url, global_lime_path, global_shap_path, global_lime_lst, global_shap_lst
    except Exception as e:
        print(e)
        return ("", "") + (None,) * 4


@app.callback(
    Output("global-lime", "src"),
    Output("global-shap", "src"),
    Output('bm_lime_path_st', 'data'),
    Output('bm_shap_path_st', 'data'),
    Output('bm_lime_lst_st', 'data'),
    Output('bm_shap_lst_st', 'data'),
    Input('validator-input-trigger', 'data'),
    State('validator-input-file', 'data'),
    State("validator-bm-model", "data"),
)
def bm_transparency_plots(trigger, input_path, bm_path):
    if trigger:
        return get_transparency_plots(input_path, bm_path, 'bm')
    return ("", "") + (None,) * 4


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
        return get_transparency_plots(input_path, rep_path, 're')
    return ("", "") + (None,) * 4

# Run the evaluation pipeline and generate word doc report


def organise_eval_output(
    metrics,
    dist_path,
    roc_path,
    pr_path,
    auc,
    psi_score,
    psi_df,
    ks,
    csi_list,
    csi_dict,
    ginis,
    lime_path,
    shap_path,
    lime_lst,
    shap_lst,
):
    psi_df = pd.read_json(psi_df, orient='split')
    csi_list = [pd.read_json(df, orient='split') for df in csi_list]
    eval_outputs = {}
    eval_outputs['Test'] = {}
    eval_outputs['Test']['charts'] = {
        'dist': dist_path,
        'pr': pr_path,
        'roc': roc_path,
        'global_lime': lime_path,
        'global_shap': shap_path,
    }
    eval_outputs['Test']['txt'] = {
        'metrics': metrics,
        'feature_gini': ginis,
        'auc': auc,
        'global_lime': lime_lst,
        'global_shap': shap_lst,
        'psi': psi_score,
        'psi_df': psi_df,
        'csi_list': csi_list,
        'csi_dict': csi_dict,
        'ks': ks
    }
    return eval_outputs


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

    State('bm_metrics_st', 'data'),
    State('bm_dist_path_st', 'data'),
    State('bm_roc_path_st', 'data'),
    State('bm_pr_path_st', 'data'),
    State('bm_auc_st', 'data'),
    State('bm_psi_score_st', 'data'),
    State('bm_psi_df_st', 'data'),
    State('bm_ks_st', 'data'),
    State('bm_ft_gini_st', 'data'),
    State('bm_lime_path_st', 'data'),
    State('bm_shap_path_st', 'data'),
    State('bm_lime_lst_st', 'data'),
    State('bm_shap_lst_st', 'data'),
    State("store-project", "data"),
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
    bm_metrics,
    bm_dist_path,
    bm_roc_path,
    bm_pr_path,
    bm_auc,
    bm_psi_score,
    bm_psi_df,
    bm_ks,
    bm_ginis,
    bm_lime_path,
    bm_shap_path,
    bm_lime_lst,
    bm_shap_lst,
    project_config,
):
    if n_clicks:
        re_eval_outputs = organise_eval_output(
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
        )
        bm_eval_outputs = organise_eval_output(
            bm_metrics,
            bm_dist_path,
            bm_roc_path,
            bm_pr_path,
            bm_auc,
            bm_psi_score,
            bm_psi_df,
            bm_ks,
            [],
            {},
            bm_ginis,
            bm_lime_path,
            bm_shap_path,
            bm_lime_lst,
            bm_shap_lst,
        )
        proj_dict = json.loads(project_config)
        proj_name = proj_dict['Project Name']
        date = proj_dict['Date']
        generate_report(
            re_eval_outputs,
            bm_eval_outputs,
            report_path=f'./outputs/{proj_name}/{date}/report.docx')
        return "Report generated!"
