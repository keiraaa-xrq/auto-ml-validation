"""Replicate model training
"""
from typing import *
import pandas as pd
import numpy as np
from .utils import instantiate_clf


def train_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_others: List[pd.DataFrame],
    algo: str,
    params: Dict,
    save: bool = False,
    save_path: Optional[str] = ''
) -> Tuple[np.ndarray, List[np.ndarray]]:
    '''
    Main function for replicating trained model and 
    output predicted probability for the positive class.
    '''
    # prepare
    clf = instantiate_clf(algo, params)
    # train
    clf.fit(X_train, y_train)
    # predict
    proba_train = clf.predict_proba(X_train)[:, 1]
    proba_others = []
    for X in X_others:
        proba = clf.predict_proba(X)[:, 1]
        proba_others.append(proba)
    # save model
    if save:
        clf.save_model(save_path)
    return proba_train, proba_others
