import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from typing import *
from imblearn.over_sampling import SMOTE
from .base_binary_classifier import BaseBinaryClassifier


class KNNClassifier(BaseBinaryClassifier):

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        save: bool = True,
        save_path: Optional[str] = "../models/knn.pkl"
    ):
        """
        Build a KNN classifier from training data.
        """
        # perfrom oversampling with SMOTE
        X_resampled, y_resampled = SMOTE(
            random_state=42, sampling_strategy=0.5).fit_resample(X_train, y_train)
        size = X_resampled.shape[0]
        self._model = KNeighborsClassifier(
            n_neighbors=int(np.sqrt(size)), weights='distance')
        super().fit(X_resampled, y_resampled, save, save_path)
