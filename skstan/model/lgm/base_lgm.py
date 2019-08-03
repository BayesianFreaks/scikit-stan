from abc import ABCMeta, abstractmethod

from skstan.model import BaseEstimator


class BaseLinearRegression(BaseEstimator, metaclass=ABCMeta):
    """
    Abstract base class for all linear regression models.

    """

    @abstractmethod
    def fit(self, X, y):
        """

        Parameters
        ----------
        X: {array-like, sparse matrix}
            Training vector.

        y: array-like
            Target vector relative to X.

        Returns
        -------

        """
        pass

    @abstractmethod
    def predict(self, X):
        """

        Parameters
        ----------
        X: {array-like, sparse matrix}

        Returns
        -------
        T: array-like
            Returns the probability.

        """
        pass
