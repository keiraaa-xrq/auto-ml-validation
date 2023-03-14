from typing import *
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from .abstract_binary_classifier import AbstractBinaryClassifier


class DTClassifier(AbstractBinaryClassifier):

    PARAM_DISTRIBUTIONS = {
        'criterion': ['gini', 'entropy'],
        'max_depth': np.arange(4, 12, 2),
        'min_samples_split': np.arange(2, 8, 2),
        'min_samples_leaf': np.arange(1, 5, 1),
    }

    def __init__(
        self,
        params: Optional[Dict] = {'random_state': 42},
        verbose: Optional[int] = 1
    ):
        super().__init__(verbose)
        self._model = DecisionTreeClassifier(**params)
        self._name = 'Decision Tree Classifier'
