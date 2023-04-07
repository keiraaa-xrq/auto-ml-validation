from auto_ml_validation.app.index import app
from auto_ml_validation.app.pages.results import *
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import json
from ...validation_package.evaluation_pipeline import evaluation_pipeline
from ...validation_package.report.generate_report import generate_report

# Layout
re_layout = html.Div(children=[
    html.H2('Result for Replicated Model',  style={
            'text-align': 'center',
            'font-weight': 'bold',
            'font-size': '30px',
            'margin-top': '20px',
            'margin-bottom': '10px',
            'text-transform': 'uppercase',
            'letter-spacing': '1px',
            'color': '#333333'}),
    html.Br(),
    re_performance_metric_layout(),
    re_statistical_model_metrics_layout(),
    re_gini_layout(),
    re_csi_table_layout(),
    re_trans_layout(),
], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top'})

bm_layout = html.Div(children=[
    html.H2('Result for Benchmark Model',  style={
            'text-align': 'center',
            'font-weight': 'bold',
            'font-size': '30px',
            'margin-top': '20px',
            'margin-bottom': '10px',
            'text-transform': 'uppercase',
            'letter-spacing': '1px',
            'color': '#333333'
            }),
    html.Br(),
    bm_performance_metric_layout(),
    bm_statistical_model_metrics_layout(),
    bm_gini_layout(),
    bm_csi_table_layout(),
    bm_trans_layout(),
], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top'})

results_layout = html.Div(children=[
    download_report_layout(),
    html.Br(),
    html.Div([
        re_layout,
        bm_layout,
    ], style={'display': 'flex'}),
    dcc.Store(id='re-result', data = {}, storage_type ='session'),
    dcc.Store(id='bm-result', data = {}, storage_type = 'session'),
    dcc.Store(id='report-trigger', data = False, storage_type = 'session'),
])

# Callbacks
# Output generic performance metrics for both Model Replication and Auto-Benchmark


@app.callback(
    [Output('dist-curve', 'figure'), Output('roc-curve', 'figure'), Output('pr-curve', 'figure'), Output('metrics', 'children'),
     Output('dist-curve-re', 'figure'), Output('roc-curve-re', 'figure'), Output('pr-curve-re', 'figure'), Output('metrics-re', 'children')],
    [Input('threshold', 'value'), Input('threshold-re', 'value'),
     Input('validator-input-trigger', 'data'),
     Input('validator-input-file', 'data')]
)
def generate_performance_metrics(threshold_bm, threshold_re, trigger, file_name):
    if trigger:
        with open(f'././data/validator_input/{file_name}', 'rb') as f:
            data = pickle.load(f)
        # Benchmark Output
        bm_test_data = data['bm_other_data']['Test']
        file_name_split = file_name.split('_')

        with open(f'././models/{file_name_split[0]}_auto_{file_name_split[2]}.pkl', 'rb') as f:
            model = pickle.load(f)
        pme = performance_metrics_evaluator.PerformanceEvaluator(bm_test_data['pred_proba'],
                                                                 float(
                                                                     threshold_bm),
                                                                 bm_test_data['y'],
                                                                 bm_test_data['processed_X'],
                                                                 model,)

        dist_bm = pme.get_dist_plot()
        roc_bm = pme.get_roc_curve()
        pr_bm = pme.get_pr_curve()
        # lift = pme.get_lift_chart()
        metrics = pme.cal_metrics()
        # confusion_matrix = pme.get_confusion_matrix()

        metrics_comp = html.Div([
            html.H6(f'Accuracy {metrics["accuracy"]}'),
            html.H6(f'Precision {metrics["precision"]}'),
            html.H6(f'Recall {metrics["recall"]}'),
            html.H6(f'F1-Score {metrics["f1_score"]}'),
        ])

        # Replication Output
        re_test_data = data['re_other_data']['Test']
        file_name_split = file_name.split('_')

        with open(f'././models/{file_name_split[0]}_{file_name_split[1]}_rep_{file_name_split[2]}.pkl', 'rb') as f:
            model = pickle.load(f)
        pme = performance_metrics_evaluator.PerformanceEvaluator(re_test_data['pred_proba'],
                                                                 float(
                                                                     threshold_re),
                                                                 re_test_data['y'],
                                                                 re_test_data['processed_X'],
                                                                 model)

        dist_re = pme.get_dist_plot()
        roc_re = pme.get_roc_curve()
        pr_re = pme.get_pr_curve()
        # lift = pme.get_lift_chart()
        metrics_re = pme.cal_metrics()
        # confusion_matrix = pme.get_confusion_matrix()

        metrics_comp_re = html.Div([
            html.H6(f'Accuracy {metrics_re["accuracy"]}'),
            html.H6(f'Precision {metrics_re["precision"]}'),
            html.H6(f'Recall {metrics_re["recall"]}'),
            html.H6(f'F1-Score {metrics_re["f1_score"]}'),
        ])
        return dist_bm, roc_bm, pr_bm, metrics_comp, dist_re, roc_re, pr_re, metrics_comp_re

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

# Output PSI and KSI


@app.callback(
    Output("psi-table", "data"), Output("psi-score",
                                        "text"), Output('psi-table', 'columns'), Output("ks-tests", "children"),
    Output("psi-table-re", "data"), Output("psi-score-re",
                                           "text"), Output('psi-table-re', 'columns'), Output("ks-tests-re", "children"),
    Input("psi-num-of-bins", "value"),
    Input('validator-input-trigger', 'data'),
    Input('validator-input-file', 'data')
)
def output_psi_ks_table(num_of_bins, trigger, file_name):
    if trigger:
        with open(f'././data/validator_input/{file_name}', 'rb') as f:
            data = pickle.load(f)
        bm_train_data = data['bm_train_data']
        bm_test_data = data['bm_other_data']['Test']
        re_train_data = data['re_train_data']
        re_test_data = data['re_other_data']['Test']
        # Auto-Benchmarking
        my_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(bm_train_data,
                                                                             bm_test_data)
        psi_score, psi_df = my_class.calculate_psi(num_of_bins)
        psi_score_text = 'PSI Score: ' + str(psi_score)
        psi_df.columns = psi_df.columns.astype(str)
        psi_df = psi_df.reset_index()
        psi_df['index'] = psi_df['index'].astype(str)
        psi_df.rename(columns={'index': 'ranges'}, inplace=True)

        ks_dict = my_class.kstest()
        ks_output = [html.H6('KS Test Train: ' + str(ks_dict['Train']), style={'textAlign': 'left', 'fontWeight': 'bold'}),
                     html.H6('KS Test Test: ' + str(ks_dict['Test']), style={
                             'textAlign': 'left', 'fontWeight': 'bold'}),
                     html.H6('KS Test Train & Test: ' + str(ks_dict['Train vs Test']), style={'textAlign': 'left', 'fontWeight': 'bold'}),]
        # Model Replication
        my_class_re = statistical_metrics_evaluator.StatisticalMetricsEvaluator(re_train_data,
                                                                                re_test_data)
        psi_score_re, psi_df_re = my_class_re.calculate_psi(num_of_bins)
        psi_score_text_re = 'PSI Score: ' + str(psi_score_re)
        psi_df_re.columns = psi_df_re.columns.astype(str)
        psi_df_re = psi_df_re.reset_index()
        psi_df_re['index'] = psi_df_re['index'].astype(str)
        psi_df_re.rename(columns={'index': 'ranges'}, inplace=True)

        ks_dict_re = my_class_re.kstest()
        ks_output_re = [html.H6('KS Test Train: ' + str(ks_dict_re['Train']), style={'textAlign': 'left', 'fontWeight': 'bold'}),
                        html.H6('KS Test Test: ' + str(ks_dict_re['Test']), style={
                                'textAlign': 'left', 'fontWeight': 'bold'}),
                        html.H6('KS Test Train & Test: ' + str(ks_dict_re['Train vs Test']), style={'textAlign': 'left', 'fontWeight': 'bold'}),]
        return psi_df.to_dict('records'), psi_score_text, [{"name": col, "id": col} for col in psi_df.columns], ks_output, psi_df_re.to_dict('records'), psi_score_text_re, [{"name": col, "id": col} for col in psi_df_re.columns], ks_output_re

# Update gini features selection based on train dataset for both models


@app.callback(
    Output("gini-feature-multi-dynamic-dropdown",
           "options"), Output("gini-feature-multi-dynamic-dropdown-re", "options"),
    Input('validator-input-trigger', 'data'),
    Input('validator-input-file', 'data')
)
def update_gini(trigger, file_name):
    if trigger:
        with open(f'././data/validator_input/{file_name}', 'rb') as f:
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
    Input('validator-input-file', 'data')
)
def update_bm_gini(ft_name_list: list[str], trigger, file_name):
    if trigger:
        with open(f'././data/validator_input/{file_name}', 'rb') as f:
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
    Input("gini-feature-multi-dynamic-dropdown-re", "value"),
    Input('validator-input-trigger', 'data'),
    Input('validator-input-file', 'data')
)
def update_re_gini(ft_name_list: list[str], trigger, file_name):
    if trigger:
        with open(f'././data/validator_input/{file_name}', 'rb') as f:
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

        return gini_children

# Generate and populate csi feature metrics for both models


@app.callback(
    Output("csi-feature-multi-dynamic-dropdown",
           "options"), Output("csi-feature-multi-dynamic-dropdown-re", "options"),
    Input('validator-input-trigger', 'data'),
    Input('validator-input-file', 'data')
)
def update_gini_bm(trigger, file_name):
    if trigger:
        with open(f'././data/validator_input/{file_name}', 'rb') as f:
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
    Input('validator-input-file', 'data')
)
def update_csi_metrics_bm(feature_list, num_of_bins, trigger, file_name):
    if trigger:
        with open(f'././data/validator_input/{file_name}', 'rb') as f:
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
    Input("csi-feature-multi-dynamic-dropdown-re", "value"),
    Input("csi-num-of-bins-re", "value"),
    Input('validator-input-trigger', 'data'),
    Input('validator-input-file', 'data')
)
def update_csi_metrics_re(feature_list, num_of_bins, trigger, file_name):
    if trigger:
        with open(f'././data/validator_input/{file_name}', 'rb') as f:
            data = pickle.load(f)
        re_train_data = data['re_train_data']
        re_test_data = data['re_other_data']['Test']
        stats_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(re_train_data,
                                                                                re_test_data)

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

# Output transparency metrics for both models


@app.callback(
    Output("global-lime", "src"), Output("local-lime",
                                         "src"), Output("global-shap", "src"), Output("local-shap", "src"),
    Output("global-lime-re", "src"), Output("local-lime-re",
                                            "src"), Output("global-shap-re", "src"), Output("local-shap-re", "src"),
    Input('validator-input-trigger', 'data'),
    Input('validator-input-file', 'data')
)
def output_transparency_plots(trigger, file_name):
    if trigger:
        with open(f'././data/validator_input/{file_name}', 'rb') as f:
            data = pickle.load(f)
        file_name_split = file_name.split('_')
        with open(f'././models/{file_name_split[0]}_auto_{file_name_split[2]}.pkl', 'rb') as f:
            auto_model = pickle.load(f)
        with open(f'././models/{file_name_split[0]}_{file_name_split[1]}_rep_{file_name_split[2]}.pkl', 'rb') as f:
            re_model = pickle.load(f)

        bm_train_data = data['bm_train_data']
        re_train_data = data['re_train_data']

        evaluator = transparency_metrics_evaluator.TransparencyMetricsEvaluator(
            auto_model, bm_train_data['processed_X'])

        local_lime_fig, global_lime_fig, local_text_lime, global_text_lime = evaluator.lime_interpretability()
        global_lime_fig.savefig(
            '././auto_ml_validation/app/assets/images/global_lime_bm.png', bbox_inches='tight')
        local_lime_fig.savefig(
            '././auto_ml_validation/app/assets/images/local_lime_bm.png', bbox_inches='tight')

        local_shap_fig, global_shap_fig, local_text_shap, global_text_shap = evaluator.shap_interpretability()
        local_shap_fig.savefig(
            '././auto_ml_validation/app/assets/images/local_shap_bm.png',  bbox_inches='tight')
        global_shap_fig.savefig(
            '././auto_ml_validation/app/assets/images/global_shap_bm.png', bbox_inches='tight')
        global_lime_bm = app.get_asset_url("images/global_lime_bm.png")
        local_lime_bm = app.get_asset_url("images/local_lime_bm.png")
        global_shap_bm = app.get_asset_url("images/global_shap_bm.png")
        local_shap_bm = app.get_asset_url("images/local_shap_bm.png")

        try:
            evaluator = transparency_metrics_evaluator.TransparencyMetricsEvaluator(
                re_model, re_train_data['processed_X'])

            local_lime_fig, global_lime_fig, local_text_lime, global_text_lime = evaluator.lime_interpretability()
            global_lime_fig.savefig(
                '././auto_ml_validation/app/assets/images/global_lime_re.png', bbox_inches='tight')
            local_lime_fig.savefig(
                '././auto_ml_validation/app/assets/images/local_lime_re.png', bbox_inches='tight')
            local_shap_fig, global_shap_fig, local_text_shap, global_text_shap = evaluator.shap_interpretability()
            local_shap_fig.savefig(
                '././auto_ml_validation/app/assets/images/local_shap_re.png',  bbox_inches='tight')
            global_shap_fig.savefig(
                '././auto_ml_validation/app/assets/images/global_shap_re.png', bbox_inches='tight')
            global_lime_re = app.get_asset_url("images/global_lime_re.png")
            local_lime_re = app.get_asset_url("images/local_lime_re.png")
            global_shap_re = app.get_asset_url("images/global_shap_re.png")
            local_shap_re = app.get_asset_url("images/local_shap_re.png")
        except Exception as e:
            global_lime_re = app.get_asset_url("images/global_lime_re.png")
            local_lime_re = app.get_asset_url("images/local_lime_re.png")
            global_shap_re = app.get_asset_url("images/global_shap_re.png")
            local_shap_re = app.get_asset_url("images/local_shap_re.png")

        return global_lime_bm, local_lime_bm, global_shap_bm, local_shap_bm, global_lime_re, local_lime_re, global_shap_re, local_shap_re


# Run the evaluation pipeline and generate word doc report
@app.callback(
    Output('re-results', 'data'), Output('bm-results', 'data'),
    Input('validator-input-trigger', 'data'), Input('validator-input-file', 'data'),
    Input('threshold', 'value'), Input('threshold-re', 'value'), 
    Input("csi-feature-multi-dynamic-dropdown", "value"), Input("psi-num-of-bins", "value"), Input("csi-num-of-bins", "value"),
)
def run_evaluation_pipeline(trigger, file_name, re_thres, bm_thres, csi_selected_ft, psi_bins, csi_bins):
    if trigger:
        with open(f'././data/validator_input/{file_name}', 'rb') as f:
            output_dict = pickle.load(f)
            # Benchmark Output
            # re_train_data = data['re_train_data']
            # re_test_data = data['re_other_data']
            # bm_train_data = data['bm_other_data']
            # bm_test_data = data['bm_other_data']

        file_name_split = file_name.split('_')
        with open(f'././models/{file_name_split[0]}_auto_{file_name_split[2]}.pkl', 'rb') as f:
            bm_model = pickle.load(f)
        with open(f'././models/{file_name_split[0]}_{file_name_split[1]}_rep_{file_name_split[2]}.pkl', 'rb') as f:
            re_model = pickle.load(f)

        if 'bm_train_data' in output_dict and bm_model is None:
            raise ValueError(f'Please give the benchmark model.')
        re_eval_outputs = {}
        re_other_data = output_dict['re_other_data']
        for ds_name, test_data in re_other_data.items():
            charts, txt = evaluation_pipeline(
                re_model.model,
                output_dict['re_train_data'],
                test_data,
                re_thres,
                csi_selected_ft, psi_bins, csi_bins
            )
            re_eval_outputs[ds_name] = {}
            re_eval_outputs[ds_name]['charts'] = charts
            re_eval_outputs[ds_name]['txt'] = txt

        if 'bm_train_data' in output_dict:
            bm_eval_outputs = {}
            bm_other_data = output_dict['bm_other_data']
            for ds_name, test_data in bm_other_data.items():
                charts, txt = evaluation_pipeline(
                    bm_model.model,
                    output_dict['bm_train_data'],
                    test_data,
                    bm_thres,
                    csi_selected_ft, psi_bins, csi_bins
                )
                bm_eval_outputs[ds_name] = {}
                bm_eval_outputs[ds_name]['charts'] = charts
                bm_eval_outputs[ds_name]['txt'] = txt
        else:
            bm_eval_outputs = None
        

        return re_eval_outputs, bm_eval_outputs


@app.callback(
    Output('report-trigger', 'data'),
    Input('download-report', 'n_clicks'), Input('re-results', 'data'), Input('bm-results', 'data')
)
def download_report(n_clicks, re_results, bm_results):
    if n_clicks:
        generate_report(re_results, bm_results)
        return True
    else:
        return False