from abc import ABCMeta, abstractmethod

from skstan.backend.stan.model.base_model import StanClassifierMixin
from skstan.backend.stan.model.base_model import StanModelLoadMixin


class BaseLinearRegressionModel(metaclass=ABCMeta):
    """
    Base class for Linear regression using Stan.
    """

    @abstractmethod
    def get_model_file_name(self):
        pass

    @abstractmethod
    def calc_distribution(self, X, params: dict):
        pass


class StanLinearRegression(BaseLinearRegressionModel, StanClassifierMixin,
                           StanModelLoadMixin):
    """
    Linear regression using Stan.

    """

    _MODEL_FILE_NAME = 'linear_regression.pkl'
    _PARAM_NAMES = [
        'alpha',
        'beta'
    ]

    def __init__(self, chains: int, warmup: int, shrinkage: int, n_jobs: int,
                 n_itr: int, algorithm: str = 'NUTS', verbose: bool = False):
        self._chains = chains
        self._warmup = warmup
        self._shrinkage = shrinkage
        self._n_jobs = n_jobs
        self._n_itr = n_itr
        self._algorithm = algorithm
        self._verbose = verbose

        self._stanfit = None

        self._stan_model = self.load_model()

    def fit(self, X, y):
        """

        Parameters
        ----------
        X: array-kile, shape (n_samples, n_features)

        y: array, shape (n_samples,)


        Returns
        -------

        """
        data = {
            'x': X,
            'y': y,
            'N': X.shape[0],
            'F': X.shape[1],
            'shrinkage': self._shrinkage
        }

        if self._stan_model is None:
            raise ValueError('stan model is not loaded.')
        self._stanfit = self._stan_model.sampling(data=data,
                                                  iter=self._n_itr,
                                                  chains=self._chains,
                                                  n_jobs=self._n_jobs,
                                                  warmup=self._warmup,
                                                  algorithm=self._algorithm,
                                                  verbose=self._verbose)
        return self

    def get_model_file_name(self):
        """

        Returns
        -------
        str
            the name of the pickled model file.
        """
        return self.__class__._MODEL_FILE_NAME

    def calc_distribution(self, X, params: dict):
        alpha = params['alpha']
        beta = params['beta']
        return X.dot(alpha.T) + beta


class StanLogisticRegression(BaseLinearRegressionModel, StanClassifierMixin,
                             StanModelLoadMixin):
    """
    Logistic regression using Stan.

    """

    _MODEL_FILE_NAME = 'logistic_regression.pkl'
    _PARAM_NAMES = [
        'alpha',
        'beta'
    ]

    def __init__(self, chains: int, warmup: int, shrinkage: int, n_jobs: int,
                 n_itr: int, algorithm: str = 'NUTS', verbose: bool = False):
        self._stan_model = self.load_model()

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

    def calc_distribution(self, X, params: dict):
        pass


class StanPoissonRegression(BaseLinearRegressionModel, StanClassifierMixin,
                            StanModelLoadMixin):
    """
    Poisson regression using Stan.
    """

    _MODEL_FILE_NAME = 'poisson_regression.pkl'
    _PARAM_NAMES = [
        'alpha',
        'beta'
    ]

    def __init__(self, chains: int, warmup: int, shrinkage: int, n_jobs: int,
                 n_itr: int, algorithm: str = 'NUTS', verbose: bool = False):
        self._stan_model = self.load_model()

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

    def calc_distribution(self, X, params: dict):
        pass
