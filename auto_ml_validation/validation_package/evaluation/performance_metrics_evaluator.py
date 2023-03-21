import pandas as pd
import sklearn.metrics as skl
import numpy as np
import plotly.express as px
import scikitplot as skp
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Optional


class PerformanceEvaluator:
    
    def __init__(self, proba, threshold, y_true, X, model):
        self.proba = proba
        self.positive_proba = [x[1] for x in proba]
        self.threshold = threshold
        self.y_true = y_true
        self.y_pred = self.get_pred()
        self.X = X
        self.model = model

    
    def get_pred(self):
        return np.array([1 if x > self.threshold else 0 for x in self.positive_proba])
    
    def get_dist_plot(self):
        chart_df = pd.DataFrame(self.positive_proba)
        chart_df.columns = ["proba"]
        return px.histogram(chart_df, x="proba",
                            title="Prediction Distribution")
    
    def get_confusion_matrix(self):
        return ConfusionMatrixDisplay.from_predictions(self.y_true, self.get_pred(), 
                                                       cmap='Oranges')

    def cal_metrics(self) -> float:
        return {
            "accuracy": skl.accuracy_score(self.y_true, self.y_pred),
            "precision": skl.precision_score(self.y_true, self.y_pred),
            "recall": skl.recall_score(self.y_true, self.y_pred),
            "f1_score": skl.f1_score(self.y_true, self.y_pred)
            }

    def get_roc_curve(self):
        fpr, tpr, thresholds = skl.roc_curve(self.y_true, self.positive_proba)
        return px.area(
            x=fpr, y=tpr,
            title='ROC Curve',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500
        )
        
    def get_pr_curve(self):
        precision, recall, thres = skl.precision_recall_curve(self.y_true, self.positive_proba)
        return px.area(
            x=recall, y=precision,
            title='Precision-Recall Curve',
            labels=dict(x='Recall', y='Precision'),
            width=700, height=500
        )

    def cal_auc(self):
        precision, recall, thres = skl.precision_recall_curve(self.y_true, self.positive_proba)
        return {
            "ROCAUC": skl.roc_auc_score(self.y_true, self.positive_proba),
            "PRAUC": skl.auc(recall, precision)
        }

    def cal_gini(self):
        """Calculate GINI index for class and attributes"""  
        def _gini_impurity (value_counts):
            n = value_counts.sum()
            p_sum = 0
            for key in value_counts.keys():
                p_sum = p_sum  +  (value_counts[key] / n ) * (value_counts[key] / n ) 
            gini = 1 - p_sum
            return gini
        
        def _gini_attribute(attribute_name):
            attribute_values = self.X[attribute_name].value_counts()
            gini_A = 0 
            for key in attribute_values.keys():
                df_k = pd.DataFrame(self.y_true)[self.X[attribute_name] == key].value_counts()
                n_k = attribute_values[key]
                n = self.X.shape[0]
                gini_A = gini_A + (( n_k / n) * _gini_impurity(df_k))
            return gini_A
        
        result = {"CLASS": _gini_impurity(pd.DataFrame(self.y_true).value_counts())}
        for key in (self.X).columns:
            result[key] = _gini_attribute(key)
        
        return result
    
    def get_lift_chart(self):
        return skp.metrics.plot_lift_curve(self.y_true, self.proba)
        
    def get_partial_dependence(self, feature: Optional[str] = None):
        if feature is None:
            feature = self.X.columns
        n_cols = 5
        return PartialDependenceDisplay.from_estimator(self.model, self.X, features=feature, n_cols=n_cols)
        