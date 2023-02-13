import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import *
from .base_binary_classifier import BaseBinaryClassifier


class RFClassifier(BaseBinaryClassifier):
    def __init__(
        self,
        verbose: Optional[int] = 1
    ):
        super().__init__(verbose)
        self._hyperparams = {
            'n_estimators': 100,
            'criterion': 'gini',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'max_features': 'sqrt',
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0.0,
            'bootstrap': True,
            'oob_score': False,
            'n_jobs': None,
            'random_state': 42,
            'verbose': 0,
            'warm_start': False,
            'class_weight': None,
            'ccp_alpha': 0.0,
            'max_samples': None
        }
        self._name = 'Random Forest Classifier'

    def _init_model(self, hyperparams_dict: Optional[Dict] = None):
        super()._init_model(hyperparams_dict)
        self._model = RandomForestClassifier(**self._hyperparams)
