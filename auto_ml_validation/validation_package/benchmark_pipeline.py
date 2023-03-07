from typing import *
import time
import pandas as pd
from joblib import Parallel, delayed
from .algorithms.abstract_binary_classifier import AbstractBinaryClassifier
from .algorithms.decision_tree import DTClassifier
from .algorithms.knn import KNNClassifier
from .algorithms.logistic_regression import LogisticClassifier
from .algorithms.random_forest import RFClassifier
from .algorithms.xgboost import XGBoostClassifier
from .algorithms.support_vector_machine import SVClassifier


def fit_model(
    clf: AbstractBinaryClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str,
    n_jobs: int,
) -> Tuple[AbstractBinaryClassifier, float, float, float]:
    """
    Perform hyperparameter tuning and threshold optimisation.
    """
    start = time.time()
    clf.random_search(X_train, y_train, metric, n_jobs=n_jobs, verbose=0)
    best_threshold, max_score = clf.optimise_threshold(
        X_val, y_val, metric, verbose=0)
    end = time.time()
    dur = (end - start) / 60
    return clf, best_threshold, max_score, dur


def instantiate_clfs(n_sample: int) -> List[AbstractBinaryClassifier]:
    """
    Create a list of blank classifier instances.
    """
    dt = DTClassifier()
    knn = KNNClassifier()
    lg = LogisticClassifier()
    rf = RFClassifier()
    xgb = XGBoostClassifier()
    svc = SVClassifier()
    if n_sample < 10000:
        clfs = [dt, knn, lg, rf, xgb, svc]
    elif n_sample < 25000:
        clfs = [dt, knn, lg, rf, xgb]
    else:
        clfs = [dt, lg, rf, xgb]
    return clfs


def compare_performance(
    results: List[Tuple[AbstractBinaryClassifier, float, float, float]],
    metric: str,
    verbose: True
) -> Tuple[str, Dict]:
    output = {}
    best_clf_name, best_threshold, best_score = None, -1, -1
    for result in results:
        clf, threshold, score, dur = result
        if verbose:
            print(
                f'Model: {clf.name}; threshold: {threshold}; {metric}: {score}; time taken: {dur:.2f} mins.')
        output[clf.name] = {
            'model': clf,
            'best_threshold': threshold,
            f'best_{metric}': score,
            'running_time': dur
        }
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_clf_name = clf.name
    if verbose:
        print(
            f"Best model is {best_clf_name} with {metric} of {best_score} at a threshold of {best_threshold}.")
    return best_clf_name, output


def auto_benchmark(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str,
    n_jobs: int = -1,
    mode: str = 'parallel',
    verbose: bool = True
) -> Tuple[str, Dict]:
    """
    Create benchmark models and return the best performing one.
    """
    modes = {'sequential', 'parallel'}
    if mode not in modes:
        raise ValueError(
            f'Invalid mode: {mode}. Please choose from "sequential" or "parallel".')
    n_sample = X_train.shape[0]
    # instantiate clfs
    clfs = instantiate_clfs(n_sample)
    if verbose:
        print('Number of classifiers: ', len(clfs))
    if mode == 'parallel':
        results = Parallel(n_jobs=-1)(
            delayed(fit_model)(clf, X_train, y_train, X_val, y_val, metric, n_jobs) for clf in clfs
        )
    else:
        results = []
        for clf in clfs:
            result = fit_model(clf, X_train, y_train,
                               X_val, y_val, metric, n_jobs)
            results.append(result)
    best_clf_name, output = compare_performance(
        results, metric, verbose=verbose)
    return best_clf_name, output
