from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from ...metrics import misclassification_error

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, counts = np.unique(y, return_counts=True)

        self.vars_ = np.zeros((len(self.classes_), X.shape[1]))

        self.mu_ = np.zeros((len(self.classes_), X.shape[1]))

        for num, clas in enumerate(self.classes_):
            for i in range(X.shape[1]):
                self.mu_[num, i] = np.mean(X[(y == clas)][:, i])

        for i in range(len(self.classes_)):
            for j in range(X.shape[1]):
                self.vars_[i, j] = np.var(X[(y == self.classes_[i])][:, j], ddof=1)

        self.pi_ = counts / len(y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        results = np.zeros(X.shape[0])

        for index, sample in enumerate(X):
            result_likelihood = np.NINF

            for i in range(len(self.classes_)):
                sub_calc = 0
                for j in range(X.shape[1]):
                    sub_calc += -np.log(np.sqrt(self.vars_[i, j]) * 2 * np.pi) - \
                                (pow(sample[j] - self.mu_[i, j], 2)) / (2 * self.vars_[i, j])

                calc = np.log(self.pi_[i]) + sub_calc

                if calc > result_likelihood:
                    results[index] = self.classes_[i]
                    result_likelihood = calc

        return results

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        result = np.zeros((X.shape[0], len(self.classes_)))

        for index, sample in enumerate(X):

            for i in range(len(self.classes_)):

                sub_calc = 0
                for j in range(X.shape[1]):
                    sub_calc += -np.log(np.sqrt(self.vars_[i, j]) * 2 * np.pi) - \
                                (pow(sample[j] - self.mu_[i, j], 2)) / (2 * self.vars_[i, j])

                result[index, i] = np.log(self.pi_[i]) + sub_calc

        return result

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
