from abc import ABC, abstractmethod
from typing import *
import warnings
import pickle
import json
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV


class AbstractBinaryClassifier(ABC):
    """
    Base class for binary classifiers.
    """

    EVAL_METRICS = {
        'accuracy': metrics.accuracy_score,
        'f1': metrics.f1_score,
        'precision': metrics.precision_score,
        'recall': metrics.recall_score,
        'roc_auc': metrics.roc_auc_score,
    }

    PARAM_DISTRIBUTIONS = {}

    @abstractmethod
    def __init__(
        self,
        params: Optional[Dict] = {},
        verbose: Optional[int] = 1
    ):
        self._model = None
        self._name = ''
        self._verbose = verbose

    @property
    def name(self):
        return self._name

    @property
    def model(self):
        return self._model

    def _verbose_print(self, msg: str):
        """
        Print the message if in verbose mode.
        """
        if self._verbose > 0:
            print(msg)

    def load_model(self, model_path: str):
        """
        Load an existing model from path.
        """
        loaded = pickle.load(open(model_path, 'rb'))
        self._model = loaded
        self._verbose_print(f'Successfully loaded model from {model_path}.')

    def save_model(self, save_path: str):
        """
        Save self._model to a pickle file.
        """
        pickle.dump(self._model, open(save_path, 'wb'))
        self._verbose_print(f'Successfully saved model to {save_path}.')

    def get_params(self):
        """
        Get the hyperparameters of the model.
        """
        return self._model.get_params()

    def set_params(self, new_params: Dict):
        """
        Modify parameter values.
        """
        self._model.set_params(**new_params)

    def save_params(self, save_path: str):
        """
        Save hyperparameters to a json file.
        """
        with open(save_path, 'w') as f:
            json.dump(self.get_params(), f)
        self._verbose_print(
            f'Successfully saved hyperparameters to {save_path}.')

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ):
        """
        Build a model from training data with specified hyperparameter values.
        """
        self._model.fit(X_train, y_train)
        self._verbose_print("Training completed.")

    def _check_valid_metric(self, metric: str):
        """
        Check if a valid metric name is provided.
        """
        if metric not in self.EVAL_METRICS:
            raise ValueError(
                f"Invalid metric: {metric}. Please select from {list(self.EVAL_METRICS.keys())}.")

    def random_search(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        metric: str,
        param_distributions: Dict = {},
        n_iter: int = 8,
        cv: int = 5,
        n_jobs=-1,
        random_state: int = 42,
        verbose: int = 3,
    ):
        """
        Perform hyperparameter tuning with random search CV.
        """
        if not param_distributions:
            param_distributions = self.PARAM_DISTRIBUTIONS
        self._check_valid_metric(metric)
        clf = RandomizedSearchCV(
            self._model,
            param_distributions,
            n_iter=n_iter,
            scoring=metric,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        clf.fit(X_train, y_train)
        self._verbose_print(
            f"Best hyperparameters: {clf.best_params_}; best score: {clf.best_score_}.")
        self._model = clf.best_estimator_

    def predict_proba(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict class probabilities for X.
        """
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
        self._verbose_print("Predictions completed.")
        return predictions

    def optimise_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric: str,
        verbose: int = 1
    ) -> Tuple[float]:
        """
        Return the optimal prediction threshold for 
        self.model based on the specific metric.
        """

        self._check_valid_metric(metric)
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        pos_proba = self.predict_proba(X_val)[:, 1]
        if metric == 'roc_auc':
            warnings.warn(
                'ROC-AUC is threshold invariant. A threshold of 0.5 will be returned.')
            score = self.EVAL_METRICS[metric](y_val, pos_proba)
            return 0.5, score

        max_score, best_threshold = -1, -1
        for t in thresholds:
            prediction = np.array([0 if p <= t else 1 for p in pos_proba])
            score = self.EVAL_METRICS[metric](y_val, prediction)
            if verbose:
                self._verbose_print(f'Threshold: {t}; Score: {score}')
            if score > max_score:
                max_score = score
                best_threshold = t
        if verbose:
            self._verbose_print(
                f"Best threshold: {best_threshold}; best {metric}: {max_score}.")
        return best_threshold, max_score
