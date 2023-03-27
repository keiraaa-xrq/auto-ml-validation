from typing import *
import time
import json
import os
import logging
import pandas as pd
from joblib import Parallel, delayed
from .algorithms.abstract_binary_classifier import AbstractBinaryClassifier
from .algorithms.decision_tree import DTClassifier
from .algorithms.knn import KNNClassifier
from .algorithms.logistic_regression import LogisticClassifier
from .algorithms.random_forest import RFClassifier
from .algorithms.xgboost import XGBoostClassifier
from .algorithms.support_vector_machine import SVClassifier
from .feature_selection.feat_selection import AutoFeatureSelector
from .utils.np_encoder import NpEncoder
from .utils.logger import setup_logger, log_info


logger = setup_logger(logging.getLogger(__name__))


def select_features(
    clf: AbstractBinaryClassifier,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    metric: str,
    n_jobs: int,
    verbose: int = 1
) -> List[str]:
    """
    Return the names of the selected features for a clf.
    """

    feature_selector = AutoFeatureSelector(
        task="binary_classification", keep=None, method='auto')
    selected, num_selected = feature_selector.generate_best_feats(
        X_train, y_train, clf.model, metric, n_jobs, verbose=verbose)
    return selected


def fit_model(
    clf: AbstractBinaryClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str,
    feature_selection: bool,
    n_jobs: int,
    verbose: int = 1
) -> Tuple[AbstractBinaryClassifier, float, float, float, List[str]]:
    """
    Perform hyperparameter tuning and threshold optimisation.
    """
    start = time.time()
    # select features
    if verbose:
        print(f'Selecting features for {clf.name}.')
    features_selected = select_features(
        clf, X_train, y_train, metric, n_jobs=n_jobs, verbose=verbose)
    train_selected = X_train[features_selected]
    val_selected = X_val[features_selected]
    # training with hyperparameter tuning
    if verbose:
        print(
            f'Compeleted feature selection for {clf.name} in {fs_dur} mins. Start training.')
    clf.random_search(train_selected, y_train, metric,
                      n_jobs=n_jobs, verbose=verbose)
    # adjust threshold
    best_threshold, max_score = clf.optimise_threshold(
        val_selected, y_val, metric, verbose=0)
    end = time.time()
    dur = (end - start) / 60
    return clf, best_threshold, max_score, dur, features_selected


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
    elif n_sample < 20000:
        clfs = [dt, knn, lg, rf, xgb]
    else:
        clfs = [dt, lg, rf, xgb]
    return clfs


def compare_performance(
    results: List[Tuple[AbstractBinaryClassifier, float, float, float, List[str]]],
    metric: str,
    verbose: bool = True
) -> Tuple[AbstractBinaryClassifier, Dict]:
    """
    Select the best classifier based on the results from fit_model.
    """

    output = {}
    best_clf, best_threshold, best_score = None, -1, -1
    for result in results:
        clf, threshold, score, dur, features_selected = result
        if verbose:
            msg = f'Model: {clf.name}; threshold: {threshold}; {metric}: {score}; time taken: {dur:.2f} mins; number of features: {len(features_selected)}.'
            log_info(logger, msg)
        output[clf.name] = {
            'model': clf,
            'best_threshold': threshold,
            f'best_{metric}': score,
            'hyperparameters': clf.get_params(),
            'running_time': dur,
            'features_selected': features_selected
        }
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_clf = clf
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
    feature_selection: bool,
    n_jobs: int = -1,
    mode: str = 'parallel',
    verbose: bool = True,
) -> Tuple[str, Dict]:
    """
    Create benchmark models and return the best performing one.
    """
    # check input validity
    metrics = {'accuracy', 'f1', 'precision', 'recall', 'roc_auc'}
    if metric not in metrics:
        raise ValueError(
            f'Invalid metric: {metric}. Please choose from {metrics}.')
    modes = {'sequential', 'parallel'}
    if mode not in modes:
        raise ValueError(
            f'Invalid mode: {mode}. Please choose from "sequential" or "parallel".')

    n_sample = X_train.shape[0]
    # instantiate clfs
    clfs = instantiate_clfs(n_sample)
    if verbose:
        log_info(logger, f'Number of classifiers: {len(clfs)}')
    if mode == 'parallel':
        results = Parallel(n_jobs=-1)(
            delayed(fit_model)(clf, X_train, y_train, X_val, y_val, metric, feature_selection, n_jobs, verbose) for clf in clfs
        )
    else:
        results = []
        for clf in clfs:
            result = fit_model(clf, X_train, y_train, X_val,
                               y_val, metric, feature_selection, n_jobs, verbose)
            results.append(result)
    best_clf, output = compare_performance(
        results, metric, verbose=verbose)
    return best_clf, output


def save_benchmark_output(output: Dict, models_dir: str, result_path: str):
    """
    Save the trained models and benchmarking output from auto_benchmark.
    """
    # check if models_dir is a directory
    if not os.path.isdir(models_dir):
        raise OSError(f'{models_dir} is not a valid directory.')

    results_dict = {}
    for name, result in output.items():
        result = result.copy()
        clf = result.pop('model')
        clf.save_model(f'{models_dir}/{name}.pkl')
        results_dict[name] = result
    with open(result_path, 'w') as fp:
        json.dump(results_dict, fp)
