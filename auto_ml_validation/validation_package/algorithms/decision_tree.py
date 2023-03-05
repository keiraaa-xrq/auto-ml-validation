import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from typing import *
from .base_binary_classifier import BaseBinaryClassifier


class DTClassifier(BaseBinaryClassifier):

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        save: bool = True,
        save_path: Optional[str] = "../models/decision_tree.pkl"
    ):
        """
        Build a decision tree classifier from training data.
        """

        self._model = DecisionTreeClassifier(
            max_depth=10, min_samples_split=5, max_features='sqrt', random_state=42, class_weight='balanced')
        super().fit(X_train, y_train, save, save_path)
