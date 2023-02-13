import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from typing import *
from .base_binary_classifier import BaseBinaryClassifier


class DTClassifier(BaseBinaryClassifier):

    def __init__(
        self,
        verbose: Optional[int] = 1
    ):
        super().__init__(verbose)
        self._hyperparams = {
            'criterion': 'gini',
            'splitter': 'best',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'max_features': None,
            'random_state': 42,
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0.0,
            'class_weight': None,
            'ccp_alpha': 0.0
        }
        self._name = 'Decision Tree Classifier'

    def _init_model(self, hyperparams_dict: Optional[Dict] = None):
        super()._init_model(hyperparams_dict)
        self._model = DecisionTreeClassifier(**self._hyperparams)
