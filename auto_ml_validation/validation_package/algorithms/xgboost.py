from typing import *
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from scipy import stats
from .base_binary_classifier import BaseBinaryClassifier


class XGBoostClassifier(BaseBinaryClassifier):

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
