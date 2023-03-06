from typing import *
import pandas as pd
from .algorithms.base_binary_classifier import BaseBinaryClassifier
from .algorithms.decision_tree import DTClassifier
from .algorithms.knn import KNNClassifier
from .algorithms.logistic_regression import LogisticClassifier
from .algorithms.random_forest import RFClassifier
from .algorithms.support_vector_machine import SVClassifier
from .algorithms.xgboost import XGBoostClassifier


def check_columns(data: pd.DataFrame, columns: List[str]):
    """
    Check the presence of columns in the data.
    """
    cols_all = set(data.columns.tolist())
    all_present = True
    not_present = []
    for c in columns:
        if c not in cols_all:
            all_present = False
            not_present.append(c)
    if not all_present:
        raise KeyError(f"Columns {not_present} are not present in the data.")


def instantiate_clf(
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
    return clf
