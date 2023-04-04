from typing import *
import pandas as pd
import sklearn.metrics as skl
import numpy as np
import plotly.express as px
import scikitplot as skp
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import BaseEstimator


class PerformanceEvaluator:

    def __init__(
        self,
        proba: np.ndarray,
        threshold: float,
        y_true: np.ndarray,
        X: pd.DataFrame,
        model: BaseEstimator
    ):
        """
        Evaluate generic model performance
        Input:
            - proba: predicted probability for both classes on testing dataset
            - threshold: classification threshold
            - y_true: true lable of testing dataset
            - X: processed features
            - model: the machine learning model used for generating prediction
        """
        self.proba = proba
        self.positive_proba = proba[:, 1]
        self.threshold = threshold
        self.y_true = y_true
        self.y_pred = self.get_pred()
        self.X = X
        self.model = model

    def get_pred(self):
        """Get predicted label based on proba and threshold"""
        return np.array([1 if x > self.threshold else 0 for x in self.positive_proba])

    def get_dist_plot(self):
        chart_df = pd.DataFrame(self.positive_proba)
        chart_df.columns = ["postive probability"]
        fig = px.histogram(chart_df, x="postive probability",
                           title="Prediction Distribution")
        # fig.show()
        return fig

    def get_confusion_matrix(self):
        return ConfusionMatrixDisplay.from_predictions(self.y_true, self.y_pred,
                                                       cmap='Oranges')

    def cal_metrics(self) -> dict[str,float]:
        return {
            "accuracy": skl.accuracy_score(self.y_true, self.y_pred),
            "precision": skl.precision_score(self.y_true, self.y_pred),
            "recall": skl.recall_score(self.y_true, self.y_pred),
            "f1_score": skl.f1_score(self.y_true, self.y_pred)
        }

    def get_roc_curve(self):
        fpr, tpr, thres = skl.roc_curve(self.y_true, self.positive_proba)
        fig = px.area(
            x=fpr, y=tpr,
            title='ROC Curve',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500
        )
       # fig.show()
        return fig

    def get_pr_curve(self):
        precision, recall, thres = skl.precision_recall_curve(
            self.y_true, self.positive_proba)
        fig = px.area(
            x=recall, y=precision,
            title='Precision-Recall Curve',
            labels=dict(x='Recall', y='Precision'),
            width=700, height=500
        )
        # fig.show()
        return fig

    def cal_auc(self):
        precision, recall, thres = skl.precision_recall_curve(
            self.y_true, self.positive_proba)
        return {
            "ROCAUC": skl.roc_auc_score(self.y_true, self.positive_proba),
            "PRAUC": skl.auc(recall, precision)
        }
    
    def get_lift_chart(self):
        return skp.metrics.plot_lift_curve(self.y_true, self.proba)


def evaluate_performance(
    pred_proba: np.ndarray,
    y_true: np.ndarray,
    X: pd.DataFrame,
    model: BaseEstimator,
    threshold: float = 0.5,
) -> Dict:
    # performance
    print("Evaluating model performance metrics...")
    pme = PerformanceEvaluator(
        pred_proba, threshold, y_true, X, model)
    metrics, auc = pme.cal_metrics(), pme.cal_auc()
    dist, confusion, roc, pr, lift = pme.get_dist_plot(), pme.get_confusion_matrix(
    ), pme.get_roc_curve(), pme.get_pr_curve(), pme.get_lift_chart()
    stats_output = {"metrics": metrics, "auc": auc}
    charts = {"dist": dist, "lift": lift, "pr": pr,
              "roc": roc, "confusion": confusion, }
    print("Model performance metrics evaluation done!")
    return stats_output, charts