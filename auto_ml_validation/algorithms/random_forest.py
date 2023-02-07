import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import *
from .base_binary_classifier import BaseBinaryClassifier


class RFClassifier(BaseBinaryClassifier):

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        save: bool = True,
        save_path: Optional[str] = "../models/random_forest.pkl"
    ):
        """
        Build a random forest classifier from training data.
        """

        self._model = RandomForestClassifier(
            max_depth=10, min_samples_split=5, max_features='sqrt', random_state=42, class_weight='balanced')
        super().fit(X_train, y_train, save, save_path)
