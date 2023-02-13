import pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import *
from .base_binary_classifier import BaseBinaryClassifier


class LogisticClassifier(BaseBinaryClassifier):
    def __init__(
        self,
        verbose: Optional[int] = 1
    ):
        super().__init__(verbose)
        self._hyperparams = {
            'penalty': 'l2',
            'dual': False,
            'tol': 1e-4,
            'C': 1.0,
            'fit_intercept': True,
            'intercept_scaling': 1,
            'class_weight': None,
            'random_state': 42,
            'solver': 'lbfgs',
            'max_iter': 100,
            'multi_class': 'auto',
            'verbose': 0,
            'n_jobs': None,
            'l1_ratio': None
        }

    def _init_model(self, hyperparams_dict: Optional[Dict] = None):
        super()._init_model(hyperparams_dict)
        self._model = LogisticRegression(**self._hyperparams)
