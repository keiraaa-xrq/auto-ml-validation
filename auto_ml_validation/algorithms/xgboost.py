import pandas as pd
from xgboost import XGBClassifier
from typing import *
from .base_binary_classifier import BaseBinaryClassifier


class XGBoostClassifier(BaseBinaryClassifier):

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        save: bool = True,
        save_path: Optional[str] = "../models/xgboost.pkl"
    ):
        """
        Build a boosting classifier from training data.
        """
        self._model = XGBClassifier()
        super().fit(X_train, y_train, save, save_path)
