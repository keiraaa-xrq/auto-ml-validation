import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from typing import *
from .base_binary_classifier import BaseBinaryClassifier


class XGBoostClassifier(BaseBinaryClassifier):

    def __init__(
        self,
        verbose: Optional[int] = 1
    ):
        super().__init__(verbose)
        self._hyperparams = {
            'objective': 'binary:logistic',
            'use_label_encoder': False,
            'base_score': None,
            'booster': 'gbtree',
            'colsample_bylevel': None,
            'colsample_bynode': None,
            'colsample_bytree': None,
            'early_stopping_rounds': None,
            'enable_categorical': False,
            'eval_metric': None,
            'gamma': None,
            'gpu_id': None,
            'grow_policy': None,
            'importance_type': None,
            'interaction_constraints': None,
            'learning_rate': None,
            'max_bin': None,
            'max_cat_to_onehot': None,
            'max_delta_step': None,
            'max_depth': None,
            'max_leaves': None,
            'min_child_weight': None,
            'missing': np.nan,
            'monotone_constraints': None,
            'n_estimators': 100,
            'n_jobs': None,
            'num_parallel_tree': None,
            'predictor': None,
            'random_state': None,
            'reg_alpha': None,
            'reg_lambda': None,
            'sampling_method': None,
            'scale_pos_weight': None,
            'subsample': None,
            'tree_method': None,
            'validate_parameters': None,
            'verbosity': None
        }
        self._name = 'XGBoost Classifier'

    def _init_model(self, hyperparams_dict: Optional[Dict] = None):
        super()._init_model(hyperparams_dict)
        self._model = XGBClassifier(**self._hyperparams)
