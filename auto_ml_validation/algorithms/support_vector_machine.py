import pandas as pd
from sklearn.svm import SVC
from typing import *
from .base_binary_classifier import BaseBinaryClassifier


class SVClassifier(BaseBinaryClassifier):
    def __init__(
        self,
        verbose: Optional[int] = 1
    ):
        super().__init__(verbose)
        self._hyperparams = {
            'C': 1.0,
            'kernel': 'rbf',
            'degree': 3,
            'gamma': 'scale',
            'coef0': 0.0,
            'shrinking': True,
            'probability': False,
            'tol': 1e-3,
            'cache_size': 200,
            'class_weight': None,
            'verbose': False,
            'max_iter': -1,
            'decision_function_shape': 'ovr',
            'break_ties': False,
            'random_state': 42
        }
        self._name = 'Support Vector Classifier'

    def _init_model(self, hyperparams_dict: Optional[Dict] = None):
        super()._init_model(hyperparams_dict)
        self._model = SVC(**self._hyperparams)
