from typing import *
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from scipy import stats
from .abstract_binary_classifier import AbstractBinaryClassifier


class XGBoostClassifier(AbstractBinaryClassifier):

    PARAM_DISTRIBUTIONS = {
        'learning_rate': stats.loguniform(a=0.005, b=0.295),
        'n_estimators': np.arange(50, 500, 50),
        'max_depth': np.arange(4, 12, 2),
        'min_child_weight': stats.uniform(loc=1, scale=9),
        'gamma': stats.uniform(loc=0, scale=0.5),
        'subsample': stats.uniform(loc=0.5, scale=0.5),
        'colsample_bytree': stats.uniform(loc=0.5, scale=0.5)
    }

    def __init__(
        self,
        params: Optional[Dict] = {'random_state': 42},
        verbose: Optional[int] = 1
    ):
        super().__init__(verbose)
        self._model = XGBClassifier(**params)
        self._name = 'XGBoost Classifier'

    def load_model(self, model_path: str):
        """
        Load an existing model from path.
        """
        self._model.load_model(model_path)
        self._verbose_print(f'Successfully loaded model from {model_path}.')

    def save_model(self, save_path: str):
        """
        Save self._model to a pickle file.
        """
        self._model.save_model(save_path)
        self._verbose_print(f'Successfully saved model to {save_path}.')
