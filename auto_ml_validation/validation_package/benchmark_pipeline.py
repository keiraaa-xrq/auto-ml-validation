import pandas as pd
from joblib import Parallel, delayed
from .algorithms.base_binary_classifier import BaseBinaryClassifier
from .algorithms.decision_tree import DTClassifier
from .algorithms.knn import KNNClassifier
from .algorithms.logistic_regression import LogisticClassifier
from .algorithms.random_forest import RFClassifier
from .algorithms.xgboost import XGBoostClassifier
from .algorithms.support_vector_machine import SVClassifier


def fit_model(
    clf: BaseBinaryClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str,
):
    clf.random_search(X_train, y_train)
    best_threshold, max_score = clf.optimise_threshold(X_val, y_val, metric)
    return best_threshold, max_score


def auto_benchmark(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str,
):
    n_sample, n_features = X_train.shape
    # instantiate clfs
    dt = DTClassifier()
    knn = KNNClassifier()
    lg = LogisticClassifier()
    rf = RFClassifier()
    xgb = XGBoostClassifier()
    svc = SVClassifier()
    # do not use SVM if smaple size > 10,000
    if n_sample < 10000:
        clfs = [dt, knn, lg, rf, xgb, svc]
    else:
        clfs = [dt, knn, lg, rf, xgb]
    # train models in parallel
    results = Parallel(n_jobs=-1)(
        delayed(fit_model)(clf, X_train, y_train, X_val, y_val, metric) for clf in clfs
    )
    output = {}
    best_index, best_threshold, best_score = -1, -1, -1
    for i, tup in enumerate(results):
        clf = clfs[i]
        threshold, score = tup
        output[clf.name] = {
            'model': clf,
            'best_threshold': threshold,
            f'best_{metric}': score,
        }
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_index = i
    best_clf = clfs[best_index]
    print(
        f"Best model is {best_clf.name} with {metric} of {best_score} at a threshold of {best_threshold}.")
    return best_clf, output
