from abc import ABC, abstractmethod
from typing import *
import pickle
import pandas as pd
import numpy as np
from sklearn import metrics


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
        self.verbose_print("Training completed.")
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

    def optimise_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric: str
    ) -> Tuple[float]:
        """
        Return the optimal prediction threshold for self.model.
        """
        metrics_dic = {
            'accuracy': metrics.accuracy_score,
            'f1': metrics.f1_score,
            'precision': metrics.precision_score,
            'recall': metrics.recall_score
        }
        if metric not in metrics_dic:
            raise ValueError(
                f"Invalid metric: {metric}. Please select from {list(metrics_dic.keys())}.")
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        pos_proba = self.predict_proba(X_val)[:, 1]
        max_score, best_threshold = -1, -1
        for t in thresholds:
            prediction = np.array([0 if p <= t else 1 for p in pos_proba])
            score = metrics_dic[metric](y_val, prediction)
            if score > max_score:
                max_score = score
                best_threshold = t
        self.verbose_print(
            f"Best threshold: {best_threshold}; best {metric}: {max_score}.")
        return best_threshold, max_score
