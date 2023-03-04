import sklearn.metrics as skl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import scikitplot as skp
from sklearn.inspection import (partial_dependence, PartialDependenceDisplay)
import matplotlib.pyplot as plt


class PerformanceEvaluator:
    def __init__(self, proba, threshold, y_true, X, model):
        self.proba = proba
        self.threshold = threshold
        self.y_true = y_true
        self.y_pred = self.get_pred()
        self.X = X
        self.model = model
    
    def get_pred(self):
        return np.array([1 if x > self.threshold else 0 for x in self.proba])
    
    def get_dist_plot(self):
        chart_df = pd.DataFrame(self.proba)
        chart_df.columns = ["proba"]
        return px.histogram(chart_df, x="proba",
                            title="Prediction Distribution")
    
    def get_confusion_matrix(self):
        return skl.confusion_matrix(self.y_true, self.y_pred)

    def cal_metrics(self) -> float:
        return {
            "accuracy": skl.accuracy_score(self.y_true, self.y_pred),
            "precision": skl.precision_score(self.y_true, self.y_pred),
            "recall": skl.recall_score(self.y_true, self.y_pred),
            "f1_score": skl.f1_score(self.y_true, self.y_pred)
            }


    def get_roc_curve(self):
        fpr, tpr, thresholds = skl.roc_curve(self.y_true, self.proba)
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        
    def get_pr_curve(self):
        precision, recall, thres = skl.precision_recall_curve(self.y_true, self.proba)
        plt.plot(recall, precision, label="Precison-recall curve")


    def get_auc(self):
        fpr, tpr, thresholds = skl.roc_curve(self.y_true, self.proba)
        return skl.auc(fpr, tpr)


    def cal_normalized_gini(self):
        """Simple normalized Gini based on Scikit-Learn's roc_auc_score"""
        gini = lambda a, p: 2 * skl.roc_auc_score(a, p) - 1
        return gini(self.y_true, self.y_pred) / gini(self.y_true, self.y_true)
    
    def get_lift_chart(self):
        skp.metrics.plot_lift_curve(
            self.y_true, self.X, figsize=(12, 8), title_fontsize=20, text_fontsize=18)
        plt.show()
        
    def get_partial_dependence(self):
        PartialDependenceDisplay.from_estimator(self.model, self.X, features=['SEX'])
