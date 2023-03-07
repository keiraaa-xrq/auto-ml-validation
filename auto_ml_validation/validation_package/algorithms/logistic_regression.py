from typing import *
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy import stats
from .abstract_binary_classifier import AbstractBinaryClassifier


class LogisticClassifier(AbstractBinaryClassifier):

    PARAM_DISTRIBUTIONS = {
        'C': stats.loguniform(a=0.01, b=100),
    }

    def __init__(
        self,
        params: Optional[Dict] = {
            'class_weight': 'balanced',
            'random_state': 42,
            'max_iter': 5000,
            'tol': 1e-3,
        },
        verbose: Optional[int] = 1
    ):
        super().__init__(verbose)
        self._model = LogisticRegression(**params)
        self._name = 'Logistic Classifier'
