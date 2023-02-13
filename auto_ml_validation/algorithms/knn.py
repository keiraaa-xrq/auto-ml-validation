import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from typing import *
from imblearn.over_sampling import SMOTE
from .base_binary_classifier import BaseBinaryClassifier


class KNNClassifier(BaseBinaryClassifier):
    def __init__(
        self,
        verbose: Optional[int] = 1
    ):
        super().__init__(verbose)
        self._hyperparams = {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto',
            'leaf_size': 30,
            'p': 2,
            'metric': 'minkowski',
            'metric_params': None,
            'n_jobs': None
        }
        self._name = 'K Neighbors Classifier'

    def _init_model(self, hyperparams_dict: Optional[Dict] = None):
        super()._init_model(hyperparams_dict)
        self._model = KNeighborsClassifier(**self._hyperparams)
