"""Replicate model training
"""

from typing import *
import logging
import pandas as pd
import numpy as np
from .utils.utils import instantiate_clf
from .algorithms.abstract_binary_classifier import AbstractBinaryClassifier
from .utils.logger import setup_logger, log_info


logger = setup_logger(logging.getLogger(__name__))


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    algo: str,
    params: Dict,
    metric: str,
    save: bool = False,
    save_path: Optional[str] = '',
    verbose: bool = True
) -> Tuple[AbstractBinaryClassifier, float]:
    '''
    Main function for replicating trained model and
    optimise the prediction threshold.
    '''
    # prepare
    clf = instantiate_clf(algo, params)
    # train
    clf.fit(X_train, y_train)
    # optimise threshold
    best_threshold, max_score = clf.optimise_threshold(
        X_val, y_val, metric, verbose=0)
    # save model
    if save:
        clf.save_model(save_path)
    if verbose:
        print(
            f'Completed training {clf.name}; best threshold: {best_threshold}; best {metric}: max_score')
    return clf, best_threshold
