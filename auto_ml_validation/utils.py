from typing import *
import pandas as pd
from sklearn import metrics
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


def evaluate(y: List[int], y_pred: List[int], class_names: List[int] = [0, 1], verbose=True):
    """
    Return evaluation metrics and print the confusion matrix.
    """

    # generate confusion matrix
    cf_mat = metrics.confusion_matrix(y, y_pred)
    row_names = ['Actual ' + str(c) for c in class_names]
    col_names = ['Predicted ' + str(c) for c in class_names]
    cf_mat = pd.DataFrame(cf_mat, index=row_names, columns=col_names)

    # calculate performance metrics
    accuracy = metrics.accuracy_score(y, y_pred)
    precision = metrics.precision_score(y, y_pred)
    recall = metrics.recall_score(y, y_pred)
    f1 = metrics.f1_score(y, y_pred)

    output_dict = {}
    output_dict['Accuracy'] = accuracy
    output_dict['Precision'] = precision
    output_dict['Recall'] = recall
    output_dict['F1'] = f1

    #print("Debug1:") ###
    if verbose:
        print(cf_mat)
        for k, v in output_dict.items():
            #print("Debug2:") ###
            print(k, ': ', v)

    return cf_mat, output_dict
