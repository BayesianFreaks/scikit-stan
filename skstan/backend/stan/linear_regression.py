from abc import ABCMeta
from abc import abstractmethod

from skstan.backend.stan import BaseStanModel


class BaseStanLinearRegression(BaseStanModel, metaclass=ABCMeta):
    """
    Abstract base class for Linear regression using Stan.

    """

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class StanLinearRegression(BaseStanLinearRegression):
    """
    Linear regression.

    """

    MODEL_FILE_NAME = 'linear_regression.pkl'

    def __init__(self):
        self._model = self.load_model()

    def fit(self, X, y):
        """

        Parameters
        ----------
        X
        y

        Returns
        -------

        """
        pass

    def predict(self, X):
        """

        Parameters
        ----------
        X

        Returns
        -------

        """
        pass

    def get_model_file_name(self):
        """

        Returns
        -------
        str
            the name of the pickled model file.
        """
        return self.__class__.MODEL_FILE_NAME


class StanLogisticRegression(BaseStanLinearRegression):
    """
    Logistic regression.

    """

    MODEL_FILE_NAME = 'logistic_regression.pkl'

    def __init__(self):
        self._model = self.load_model()

    def fit(self, X, y):
        """

        Parameters
        ----------
        X
        y

        Returns
        -------
        str
            the name of the pickled model file.
        """
        pass

    def predict(self, X):
        """

        Parameters
        ----------
        X

        Returns
        -------

        """
        pass

    def get_model_file_name(self):
        """

        Returns
        -------
        str
            the name of the pickled model file.
        """
        return self.__class__.MODEL_FILE_NAME


class StanPoissonRegression(BaseStanLinearRegression):
    """
    Poisson regression.
    """

    MODEL_FILE_NAME = 'poisson_regression.pkl'

    def __init__(self):
        self._model = self.load_model()

    def fit(self, X, y):
        """

        Parameters
        ----------
        X
        y

        Returns
        -------

        """
        pass

    def predict(self, X):
        """

        Parameters
        ----------
        X

        Returns
        -------

        """
        pass

    def get_model_file_name(self):
        """

        Returns
        -------
        str
            the name of the pickled model file.
        """
        return self.__class__.MODEL_FILE_NAME
