from abc import ABC, abstractmethod
from typing import *
import pickle
import pandas as pd
import numpy as np


class BaseBinaryClassifier(ABC):
    """
    Base class for binary classifiers.
    """

    def __init__(
        self,
        verbose: Optional[int] = 1
    ):
        self._model = None
        self._verbose = verbose

    def verbose_print(self, msg):
        if self._verbose > 0:
            print(msg)

    def load_model(self, model_path):
        """
        Load an existing model from path.
        """
        loaded = pickle.load(open(model_path, 'rb'))
        self._model = loaded
        self.verbose_print(model_path + ' successfully loaded.')

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        save: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Build a model from training data.
        """
        self._model.fit(X_train, y_train)
        if save:
            pickle.dump(self._model, open(save_path, 'wb'))

    def check_is_fitted(self):
        """
        Check if the model has been fitted.
        """
        if self._model is None:
            raise AttributeError(
                "This instance is not fitted yet. Call 'fit' or 'load_model' before using this estimator."
            )

    def predict_proba(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict class probabilities for X.
        """
        self.check_is_fitted()
        pred_proba = self._model.predict_proba(X)
        return pred_proba

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict class for X based on the classification threshold.
        """
        pos_proba = self.predict_proba(X)[:, 1]
        predictions = np.array([0 if p <= threshold else 1 for p in pos_proba])
        self.verbose_print("Predictions completed.")
        return predictions
