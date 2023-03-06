"""Replicate model training
"""

from typing import *
import pandas as pd
from .process_data import split_x_y
from .utils import instantiate_clf, evaluate


def train_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    algo: str,
    params: Dict,
    save: bool = False,
    save_path: Optional[str] = ''
):
    '''
    Main function for replicating trained model.
    '''
    # prepare
    clf = instantiate_clf(algo, params)
    # train
    print('Training')
    clf.fit(X_train, y_train)
    # predict
    print('Predict')
    proba_train = clf.predict_proba(X_train)[:, 1]
    proba_test = clf.predict_proba(X_test)[:, 1]
    # save model
    if save:
        clf.save_model(save_path)
    return proba_train, proba_test
