"""
Consolidate all the evaluations and generate word format report
"""

from typing import *
import pandas as pd
from evaluation import *


def evaluation_pipeline(
    model,
    train: Dict[str, Union[pd.DataFrame, np.ndarray]],
    test: Dict[str, Union[pd.DataFrame, np.ndarray]],
    threshold: float,
    selected_features: List[str]
):
    """
    Takes in model, data and parameters and generate one dictionary of numerical results and one dictionary for graphical results
    Input:
        train, test: dictionary with 'raw_X', 'processed_X', 'y', 'pred_proba' as keys and pandas DataFrame or numpy ndarray as values
    
    """
    # check whether the dictionary contains all the datasets needed

    # performance
    print("Evaluating model performance metrics...")
    pme = PerformanceEvaluator(test['pred_proba'], threshold, test['y'], test['processed_X'], model)
    metrics, auc = pme.cal_metrics(), pme.cal_auc()
    dist, confusion, roc, pr, lift = pme.get_dist_plot(), pme.get_confusion_matrix(), pme.get_roc_curve(), pme.get_pr_curve(), pme.get_lift_chart()
    print("Model performance metrics evaluation done!")

    # statistical
    print("Evaluating statistical metrics...")
    sme = StatisticalMetricsEvaluator(train, test)
    psi, psi_df = sme.calculate_psi()
    csi_list, csi_dict = sme.csi_for_all_features(selected_features)
    ks = sme.kstest()
    feature_gini = sme.cal_feature_gini()
    dataset_gini = sme.cal_normalized_gini()
    print("Statistical metrics evaluation done!")

    # transparency
    print("Evaluating transparency metrics...")
    tme = TransparencyMetricsEvaluator(model, train['processed_X'].iloc[0:250, :])  # for testing
    local_lime_fig, global_lime_fig, local_lime_lst, global_lime_map = tme.lime_interpretability()
    local_shap_fig, global_shap_fig, local_impt_map, global_impt_map = tme.shap_interpretability()

    print("Transparency metrics evaluation done!")

    return {
            "dist": dist, "lift": lift, "pr": pr, "roc": roc, "confusion": confusion,
            "local_lime": local_lime_fig, "global_lime": global_lime_fig, "local_shap": local_shap_fig, "global_shap": global_shap_fig, 
        }, {
            "metrics": metrics, "feature_gini": feature_gini, "auc": auc, "dataset_gini": dataset_gini, 
            "local_lime": local_lime_lst, "global_lime": global_lime_map, "local_shap": local_impt_map, "global_shap": global_impt_map,
            "psi": psi, "psi_df": psi_df, 
            "csi_list": csi_list, "csi_dict": csi_dict, "ks": ks
        }
