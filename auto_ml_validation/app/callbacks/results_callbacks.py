from auto_ml_validation.app.index import app
from auto_ml_validation.app.pages.results import *
from dash.dependencies import Input, Output, State
import json
# Body Content

# Update the PSI table when user change num of bins
@app.callback(
    Output("psi-table", "data"),
    Input("psi-num-of-bins", "value")
)
# def update_statistical_metrics(num_of_bins):
def update_psi_table(num_of_bins):
        my_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(train_data, 
                                                                         test_data)
        psi_score, psi_df = my_class.calculate_psi(num_of_bins)
        # reset index as columns for display
        psi_df.columns = psi_df.columns.astype(str)
        psi_df = psi_df.reset_index()
        psi_df['index'] = psi_df['index'].astype(str)
        psi_df.rename(columns = {'index':'ranges'}, inplace = True)

        return psi_df.to_dict('records')
        
# Update the PSI score when user change num of bins
@app.callback(
    Output("psi-score", "children"),
    Input("psi-num-of-bins", "value")
)
def update_psi_score(num_of_bins):
        my_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(train_data, 
                                                                         test_data)
        psi_score, psi_df = my_class.calculate_psi(num_of_bins)

        return 'PSI Score: '+ str(psi_score)
               
# Update the CSI when user select features
# options = train_data['raw_X'].columns.to_list()
@app.callback(
    Output("feature_related_viz", "children"),
    Input("csi-feature-multi-dynamic-dropdown", "value"),
    Input("csi-num-of-bins", "value")
)
def update_feature_related_metrcis(feature_list, num_of_bins):
        stats_class = statistical_metrics_evaluator.StatisticalMetricsEvaluator(train_data, 
                                                                                test_data)
        csi_df, csi_dict = stats_class.csi_for_all_features(feature_list, num_of_bins)

        csi_children = []
        for df, ft_name in zip(csi_df, feature_list):
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
        return html.Div(id="feature_related_viz", children = csi_children)


# Layout
results_layout = html.Div(children=[
    html.Br(),
    html.Br(),
    html.Br(),
    statistical_model_metrics_layout(train_data, test_data, 10),
    html.Br(),
    html.Br(),
    feature_metrics_layout(train_data, test_data, 
                           [],
                            10)
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

