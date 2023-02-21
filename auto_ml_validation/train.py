"""Replicate model training
"""

from typing import *
import pandas as pd
from .algorithms.base_binary_classifier import BaseBinaryClassifier
from .algorithms.decision_tree import DTClassifier
from .algorithms.knn import KNNClassifier
from .algorithms.logistic_regression import LogisticClassifier
from .algorithms.random_forest import RFClassifier
from .algorithms.support_vector_machine import SVClassifier
from .algorithms.xgboost import XGBoostClassifier


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    algo: str,
    params: Dict,
) -> BaseBinaryClassifier:
    if algo == 'dt':
        clf = DTClassifier(params)
    elif algo == 'knn':
        clf = KNNClassifier(params)
    elif algo == 'logistic':
        clf = LogisticClassifier(params)
    elif algo == 'rf':
        clf = RFClassifier(params)
    elif algo == 'svm':
        clf = SVClassifier(params)
    elif algo == 'xgboost':
        clf = XGBoostClassifier(params)
    else:
        raise ValueError(f'Invalid algo name: {algo}')
    clf.fit(X_train, y_train)
    return clf
