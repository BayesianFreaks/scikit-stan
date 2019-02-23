from skstan.backend.stan.model.base_model import BaseStanModel


class StanLinearRegression(BaseStanModel):
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


class StanLogisticRegression(BaseStanModel):
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


class StanPoissonRegression(BaseStanModel):
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
