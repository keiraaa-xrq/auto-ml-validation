"""Replicate model training
"""

from typing import *
import pandas as pd
import numpy as np
from .utils.utils import instantiate_clf


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    algo: str,
    params: Dict,
    save: bool = False,
    save_path: Optional[str] = ''
) -> Tuple[np.ndarray, List[np.ndarray]]:
    '''
    Main function for replicating trained model.
    '''
    # prepare
    clf = instantiate_clf(algo, params)
    # train
    clf.fit(X_train, y_train)
    # save model
    if save:
        clf.save_model(save_path)
    return clf
