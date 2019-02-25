from abc import ABCMeta

from skstan.backend.stan.model.base_model import StanModelLoadMixin


class BaseLinearRegressionModel(metaclass=ABCMeta):

    def fit(self, X, y):
        """

        Parameters
        ----------
        X: array-kile, shape (n_samples, n_features)

        y: array, shape (n_samples,)


        Returns
        -------

        """

        pass

    def predict(self, X):
        """

        Parameters
        ----------
        X: array-like


        Returns
        -------
        y_pred: array, shape (n_samples,)
            class labels for sample in X.

        """

        pass


class StanLinearRegression(BaseLinearRegressionModel, StanModelLoadMixin):
    """
    Linear regression using Stan.

    """

    _MODEL_FILE_NAME = 'linear_regression.pkl'

    def __init__(self):
        self._model = self.load_model()

    def _validate_params(self):
        pass

    def get_model_file_name(self):
        """

        Returns
        -------
        str
            the name of the pickled model file.
        """
        return self.__class__._MODEL_FILE_NAME


class StanLogisticRegression(BaseLinearRegressionModel, StanModelLoadMixin):
    """
    Logistic regression using Stan.

    """

    _MODEL_FILE_NAME = 'logistic_regression.pkl'

    def __init__(self):
        self._model = self.load_model()

    def _validate_params(self):
        pass

    def get_model_file_name(self):
        """

        Returns
        -------
        str
            the name of the pickled model file.
        """
        return self.__class__._MODEL_FILE_NAME


class StanPoissonRegression(BaseLinearRegressionModel, StanModelLoadMixin):
    """
    Poisson regression using Stan.
    """

    _MODEL_FILE_NAME = 'poisson_regression.pkl'

    def __init__(self):
        self._model = self.load_model()

    def _validate_params(self):
        pass

    def get_model_file_name(self):
        """

        Returns
        -------
        str
            the name of the pickled model file.
        """
        return self.__class__._MODEL_FILE_NAME
