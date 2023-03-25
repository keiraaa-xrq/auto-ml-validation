from typing import *
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.svm import SVC
from .abstract_binary_classifier import AbstractBinaryClassifier


class SVClassifier(AbstractBinaryClassifier):

    PARAM_DISTRIBUTIONS = {
        'C': stats.loguniform(a=0.01, b=100),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4],
    }

    def __init__(
        self,
        params: Optional[Dict] = {
            'class_weight': 'balanced',
            'random_state': 42,
            'probability': True,
            'tol': 1e-3
        },
        verbose: Optional[int] = 1
    ):
        super().__init__(verbose)
        self._model = SVC(**params)
        self._name = 'Support Vector Classifier'
