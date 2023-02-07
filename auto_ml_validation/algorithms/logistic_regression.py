import pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import *
from .base_binary_classifier import BaseBinaryClassifier


class LogisticClassifier(BaseBinaryClassifier):

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        save: bool = True,
        save_path: Optional[str] = "../models/logistic_regression.pkl"
    ):
        """
        Build a simple logistic regression classifier from training data.
        """
        self._model = LogisticRegression(solver='lbfgs', max_iter=1000)
        super().fit(X_train, y_train, save, save_path)
