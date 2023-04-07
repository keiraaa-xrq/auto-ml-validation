"""
Consolidate all the evaluations and generate word format report
"""

from typing import *
import pandas as pd
from .evaluation import *


def evaluation_pipeline(
    model,
    train: Dict[str, Union[pd.DataFrame, np.ndarray]],
    test: Dict[str, Union[pd.DataFrame, np.ndarray]],
    threshold: float,
    selected_features: List[str],
    psi_bin: int, csi_bin: int
):
    """
    Takes in model, data and parameters and generate one dictionary of numerical results and one dictionary for graphical results
    Input:
        train, test: dictionary with 'raw_X', 'processed_X', 'y', 'pred_proba' as keys and pandas DataFrame or numpy ndarray as values

    """
    # check whether the dictionary contains all the datasets needed

    # performance

    print("Evaluating model performance metrics...")
    pme = PerformanceEvaluator(
        test['pred_proba'], threshold, test['y'], test['processed_X'], model)
    try:
        metrics = pme.cal_metrics()
    except Exception as e:
        print(f'Error in calculating performance metrics: {e}')
        metrics = None
    try:
        auc = pme.cal_auc()
    except Exception as e:
        print(f'Error in calculating ROC AUC: {e}')
        auc = None
    try:
        dist = pme.get_dist_plot()
    except Exception as e:
        print(f'Error in ploting probaility distributions: {e}')
        dist = None
    try:
        confusion = pme.get_confusion_matrix()
    except Exception as e:
        print(f'Error in ploting confusion matrix: {e}')
        confusion = None
    try:
        roc = pme.get_roc_curve()
    except Exception as e:
        print(f'Error in ploting ROC curve: {e}')
        roc = None
    try:
        pr = pme.get_pr_curve()
    except Exception as e:
        print(f'Error in ploting Precision-Recall curve: {e}')
        pr = None
    try:
        lift = pme.get_lift_chart()
    except Exception as e:
        print(f'Error in ploting Lift chart: {e}')
        lift = None
    print("Model performance metrics evaluation done!")

    # statistical
    print("Evaluating statistical metrics...")
    sme = StatisticalMetricsEvaluator(train, test)
    try:
        psi, psi_df = sme.calculate_psi(num_bins=psi_bin)
    except Exception as e:
        print(f'Error in calculating PSI: {e}')
        psi = psi_df = None
    try:
        csi_list, csi_dict = sme.csi_for_all_features(selected_features, csi_bin)
    except Exception as e:
        print(f'Error in calculating CSI: {e}')
        csi_list = csi_dict = None
    try:
        ks = sme.kstest()
    except Exception as e:
        print(f'Error in KS test: {e}')
        ks = None
    try:
        feature_gini = sme.cal_feature_gini()
    except Exception as e:
        print(f'Error in calculating gini index for features: {e}')
        feature_gini = None
    try:
        dataset_gini = sme.cal_normalized_gini()
    except Exception as e:
        print(f'Error in calculating global gini index: {e}')
        dataset_gini = None
    print("Statistical metrics evaluation done!")

    # transparency
    print("Evaluating transparency metrics...")
    tme = TransparencyMetricsEvaluator(
        model, train['processed_X'].iloc[:100, :])
    try:
        local_lime_fig, global_lime_fig, local_lime_lst, global_lime_map = tme.lime_interpretability()
    except Exception as e:
        print(f'Error in LIME interpretability: {e}')
        local_lime_fig = global_lime_fig = local_lime_lst = global_lime_map = None
    try:
        local_shap_fig, global_shap_fig, local_impt_map, global_impt_map = tme.shap_interpretability()
    except Exception as e:
        print(f'Error in SHAP interpretability: {e}')
        local_shap_fig = global_shap_fig = local_impt_map = global_impt_map = None
    print("Transparency metrics evaluation done!")

    charts = {
        "dist": dist, "lift": lift, "pr": pr, "roc": roc, "confusion": confusion,
        "local_lime": local_lime_fig, "global_lime": global_lime_fig, "local_shap": local_shap_fig, "global_shap": global_shap_fig,
    }
    txt = {
        "metrics": metrics, "feature_gini": feature_gini, "auc": auc, "dataset_gini": dataset_gini,
        "local_lime": local_lime_lst, "global_lime": global_lime_map, "local_shap": local_impt_map, "global_shap": global_impt_map,
        "psi": psi, "psi_df": psi_df,
        "csi_list": csi_list, "csi_dict": csi_dict, "ks": ks
    }
    return charts, txt
