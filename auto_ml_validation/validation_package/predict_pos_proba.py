from typing import *
import pandas as pd
import numpy as np
from .algorithms.abstract_binary_classifier import AbstractBinaryClassifier


def predict_pos_proba(
    clf: AbstractBinaryClassifier,
    X_dfs: List[pd.DataFrame]
) -> List[np.ndarray]:
    """
    Predict the positive-class probabilities for the input dataframes.
    """
    probas = []
    for X in X_dfs:
        proba = clf.predict_proba(X)[:, 1]
        probas.append(proba)
    return probas
