from abc import ABCMeta, abstractmethod

from skstan.model import BaseEstimator


class BaseLinearRegression(BaseEstimator, metaclass=ABCMeta):
    """
    Abstract base class for all linear regression models.

    """

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
