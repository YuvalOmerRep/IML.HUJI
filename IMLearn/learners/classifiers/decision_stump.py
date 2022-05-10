from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        threshes_losses_one = np.apply_along_axis(self._find_threshold, 1, X, y, 1)
        threshes_losses_minus = np.apply_along_axis(self._find_threshold, 1, X, y, -1)

        one_best = np.argmin(threshes_losses_one[:, 1], axis=1)
        minus_best = np.argmin(threshes_losses_minus[:, 1], axis=1)

        if threshes_losses_one[one_best][1] > threshes_losses_minus[minus_best][1]:
            self.threshold_ = threshes_losses_minus[minus_best][0]
            self.j_ = threshes_losses_minus[minus_best][1]
            self.sign_ = -1
        else:
            self.threshold_ = threshes_losses_one[minus_best][0]
            self.j_ = threshes_losses_one[minus_best][1]
            self.sign_ = 1

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        feature = X[:, self.j_]
        feature[feature >= self.threshold_] = self.sign_
        feature[feature < self.threshold_] = -self.sign_
        return feature

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sort_indexes = np.argsort(values)

        values, labels = values[sort_indexes], labels[sort_indexes]

        labels_sign, labels_minus = np.abs(np.where(np.sign(labels) != 1, 0, labels)), \
                                    np.abs(np.where(np.sign(labels) == 1, 0, labels))

        labels_sign = np.insert(np.cumsum(labels_sign), 0, 0)

        best_value = 0
        best_thresh_loss = np.inf

        for index, value in enumerate(values):
            curr_loss = labels_sign[index] + np.sum(labels_minus[index:])

            if curr_loss < best_thresh_loss:
                best_thresh_loss = curr_loss
                best_value = value

        return best_value, best_thresh_loss

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.predict(X))
