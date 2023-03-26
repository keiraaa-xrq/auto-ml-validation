import pandas as pd
from typing import *
from .fairness_metrics_evaluator import *
from .performance_metrics_evaluator import *
from .statistical_metrics_evaluator import *
from .transparency_metrics_evaluator import *


class ModelEvaluator:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        raw_train: pd.DataFrame,
        raw_test: pd.DataFrame,
        class_name_list: List[str] = None
    ):
        """
        Initiate with the datasets to be used for testing
        X_train, y_train, X_test, y_test: fully processed datasets
        raw_train, raw_test: semi-processed datasets without encoding
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.raw_train = raw_train
        self.raw_test = raw_test
        self.class_name = class_name_list

    def evaluate_model(self, model, proba, threshold):
        # performance
        print("Evaluating model performance metrics...")
        pme = PerformanceEvaluator(
            proba, threshold, self.y_test, self.X_test, model)
        metrics, gini, auc = pme.cal_metrics(), pme.cal_gini(), pme.cal_auc()
        dist, confusion, roc, pr, lift, pdp = pme.get_dist_plot(), pme.get_confusion_matrix(
        ), pme.get_roc_curve(), pme.get_pr_curve(), pme.get_lift_chart(), pme.get_partial_dependence()
        print("Model performance metrics evaluation done!")

        # statistical
        # sme = StatisticalMetricsEvaluator()

        # transparency
        print("Evaluating transparency metrics...")
        tme = TransparencyMetricsEvaluator(model, self.X_train.iloc[0:2500, :])
        local_lime_fig, global_lime_fig, local_lime_lst, global_lime_map = tme.lime_interpretability()
        local_shap_fig, global_shap_fig, local_impt_map, global_impt_map = tme.shap_interpretability()
        print("Transparency metrics evaluation done!")

        return {
            "metrics": metrics, "dist": dist, "lift": lift, "pr": pr, "roc": roc, "pdp": pdp, "confusion": confusion,
            "gini": gini, "auc": auc,

            "local_lime_fig": local_lime_fig, "global_lime_fig": global_lime_fig, "local_lime_map": local_lime_lst, "global_lime_map": global_lime_map,
            "local_shap_fig": local_shap_fig, "global_shap_fig": global_shap_fig, "local_impt_map": local_impt_map, "global_impt_map": global_impt_map
        }
