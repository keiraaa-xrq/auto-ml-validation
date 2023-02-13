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

    @abstractmethod
    def __init__(
        self,
        verbose: Optional[int] = 1
    ):
        self._model = None
        self._hyperparams = {}
        self._name = ''
        self._verbose = verbose

    def verbose_print(self, msg: str):
        """
        Print the message if in verbose mode.
        """
        if self._verbose > 0:
            print(msg)

    @property
    def hyperparams(self):
        """
        Get the hyperparameters of the model.
        """
        return self._hyperparams

    def _check_hyperparams_are_valid(self, hyperparams_dict: Dict):
        """
        Check if valid hyperparameter names are provided.
        """
        for p in hyperparams_dict.keys():
            if p not in self._hyperparams:
                raise KeyError(
                    f"'{p}' is not a valid hyperparameter for {self._name}. The allowed hyperparameters are {list(self._hyperparams.keys())}")

    @hyperparams.setter
    def hyperparams(self, hyperparams_dict: Dict):
        """
        Update the hyperparameters.
        """
        self._check_hyperparams_are_valid(hyperparams_dict)
        self._hyperparams.update(hyperparams_dict)

    def load_model(self, model_path: str):
        """
        Load an existing model from path.
        """
        loaded = pickle.load(open(model_path, 'rb'))
        self._model = loaded
        self.verbose_print(model_path + ' successfully loaded.')

    @abstractmethod
    def _init_model(self, hyperparams_dict: Optional[Dict] = None):
        """
        Instantiate an empty model with the specified hyperparameters.
        """
        if hyperparams_dict is not None:
            self.hyperparams = hyperparams_dict

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hyperparams_dict: Optional[Dict] = None,
        save: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Build a model from training data with hyperparameter tuning.
        """
        self._init_model(hyperparams_dict)
        self._model.fit(X_train, y_train)
        self.verbose_print("Training completed.")
        if save:
            if save_path is None:
                raise ValueError(
                    "Please provide a valid path for saving the model.")
            pickle.dump(self._model, open(save_path, 'wb'))

    def _check_is_fitted(self):
        """
        Check if the model has been fitted.
        """
        if self._model is None:
            raise AttributeError(
                f"This {self._name} instance is not fitted yet. Call 'fit' or 'load_model' before using this estimator."
            )

    def predict_proba(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict class probabilities for X.
        """
        self._check_is_fitted()
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
