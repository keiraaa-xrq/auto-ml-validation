from typing import *
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy import stats
from .base_binary_classifier import BaseBinaryClassifier


class LogisticClassifier(BaseBinaryClassifier):

    PARAM_DISTRIBUTIONS = {
        'C': stats.loguniform(a=0.01, b=100),
        'solver': ['lbfgs', 'liblinear', 'sag'],
    }

    def __init__(
        self,
        params: Optional[Dict] = {
            'class_weight': 'balanced',
            'random_state': 42,
            'max_iter': 2000,
            'tol': 1e-4,
        },
        verbose: Optional[int] = 1
    ):
        super().__init__(verbose)
        self._model = LogisticRegression(**params)
        self._name = 'Logistic Classifier'
