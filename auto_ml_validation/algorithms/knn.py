from typing import *
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from .base_binary_classifier import BaseBinaryClassifier


class KNNClassifier(BaseBinaryClassifier):

    PARAM_DISTRIBUTIONS = {
        'n_neighbors': np.arange(5, 35, 5),
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3, 4, 5]
    }

    def __init__(
        self,
        params: Optional[Dict] = {'weights': 'distance'},
        verbose: Optional[int] = 1
    ):
        super().__init__(verbose)
        self._model = KNeighborsClassifier(**params)
        self._name = 'K Neighbors Classifier'
