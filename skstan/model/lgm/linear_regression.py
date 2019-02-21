from skstan.backend import Backend
from skstan.model.lgm import BaseLinearRegression


class LinearRegression(BaseLinearRegression):
    """
    Linear regression model class.

    Parameters
    ----------

    """

    def __init__(self):
        self._bk_model = Backend.LinearRegression()

    def _validate_params(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class LogisticRegression(BaseLinearRegression):
    """
    Logistic regression model class.

    Parameters
    ----------

    """

    def __init__(self):
        self._bk_model = Backend.LogisticRegression()

    def _validate_params(self):
        pass

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


class PoissionRegression(BaseLinearRegression):

    def __init__(self):
        self._bk_model = Backend.PoissionRegression()

    def _validate_params(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
