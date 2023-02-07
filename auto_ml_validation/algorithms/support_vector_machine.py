import pandas as pd
from sklearn.svm import SVC
from typing import *
from .base_binary_classifier import BaseBinaryClassifier


class SVClassifier(BaseBinaryClassifier):

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        save: bool = True,
        save_path: Optional[str] = "../models/svm.pkl"
    ):
        """
        Build a support vector classifier from training data.
        """
        self._model = SVC(kernel="rbf", C=0.0001,
                          gamma='scale', probability=True)
        super().fit(X_train, y_train, save, save_path)
