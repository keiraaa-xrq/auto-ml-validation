import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import *
from .abstract_binary_classifier import AbstractBinaryClassifier


class RFClassifier(AbstractBinaryClassifier):

    PARAM_DISTRIBUTIONS = {
        'n_estimators': np.arange(50, 500, 50),
        'criterion': ['gini', 'entropy'],
        'max_depth': np.arange(4, 20, 2),
        'min_samples_split': np.arange(2, 10, 2),
        'min_samples_leaf': np.arange(1, 10, 2),
    }

    def __init__(
        self,
        params: Optional[Dict] = {'random_state': 42},
        verbose: Optional[int] = 1
    ):
        super().__init__(verbose)
        self._model = RandomForestClassifier(**params)
        self._name = 'Random Forest Classifier'
