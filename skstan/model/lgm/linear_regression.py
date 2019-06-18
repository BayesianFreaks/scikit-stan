from skstan.backend import Backend
from skstan.model.lgm import BaseLinearRegression


class LinearRegression(BaseLinearRegression):
    """
    Linear regression model class.

    Parameters
    ----------
    chains: int
        the number of chains.
    shrinkage: int (default=10)
        the standard deviation for non-information prior distribution.
        Usually, the deviation is very large.

    """

    def __init__(self, chains: int, shrinkage: int = 10):
        self._backend_model = Backend.LinearRegression()
        self._validate_params(chains, shrinkage)

    @staticmethod
    def _validate_params(chains: int, shrinkage: int):
        if chains < 0:
            raise ValueError('chains must be positive.')
        if shrinkage < 0:
            raise ValueError('shrinkage must be positive.')

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
        self._backend_model.fit(X, y)
        return self

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
        return self._backend_model.predict(X)

    def predict_proba(self, X):
        pass

    def predict_log_proba(self):
        pass


class LogisticRegression(BaseLinearRegression):
    """
    Logistic regression model class.

    Parameters
    ----------

    """

    def __init__(self):
        self._backend_model = Backend.LogisticRegression()

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
        self._backend_model.fit(X, y)
        return self

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
        return self._backend_model.predict(X)


class PoissonRegression(BaseLinearRegression):

    def __init__(self):
        self._backend_model = Backend.PoissonRegression()

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
        self._backend_model.fit(X, y)
        return self

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
        return self._backend_model.predict(X)
