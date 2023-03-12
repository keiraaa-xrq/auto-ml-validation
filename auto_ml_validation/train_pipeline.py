"""Replicate model training
"""

from typing import *
import pandas as pd
from .process_data import split_x_y
from .utils import instantiate_clf, evaluate


def train_pipeline(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target: str,
    algo: str,
    params: str,
    save: bool = False,
    save_path: Optional[str] = ''
):
    '''
    Main function for replicating trained model.
    '''
    # prepare
    print('Prepare for training')
    X_train, y_train = split_x_y(train_data, target)
    X_test, y_test = split_x_y(test_data, target)
    clf = instantiate_clf(algo, params)
    # train
    print('Training')
    clf.fit(X_train, y_train)
    # predict
    print('Predict and evaluate')
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    # evaluate
    train_clf, train_metrics = evaluate(y_train, pred_train, verbose=False)
    test_clf, test_metrics = evaluate(y_test, pred_test, verbose=False)
    # save model
    if save:
        clf.save_model(save_path)
    return train_clf, train_metrics, test_clf, test_metrics
