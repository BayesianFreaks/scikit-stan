from abc import ABCMeta
from abc import abstractmethod

from skstan.backend.stan.model.base_model import StanClassifierMixin
from skstan.backend.stan.model.base_model import StanModelLoadMixin
from skstan.params import StanLinearRegressionParams


class BaseLinearRegressionModel(metaclass=ABCMeta):
    """
    Base class for Linear regression using Stan.
    """
    @abstractmethod
    def get_model_file_path(self):
        pass

    @abstractmethod
    def calc_distribution(self, X, params: dict):
        pass


class StanLinearRegression(BaseLinearRegressionModel, StanClassifierMixin,
                           StanModelLoadMixin):
    """
    Linear regression using Stan.

    Parameters
    ----------
    chains: int
        The number of chains.
    warmup: int
        Markov chains may take some time before the chain settles into the
        equilibrium distribution. The samples generated during the initial
        phase are discarded. The warmup value is the number of steps to be
        discarded.
    shrinkage: int (default=10)
        The standard deviation for non-information prior distribution.
        Usually, the deviation value is very large.
    n_itr: int
        The number of iteration.
    n_jobs: int (default=-1)
        The number of jobs to run in parallel.
    algorithm: str (default=None)
       the algorithm name to be used in Stan.
    verbose: bool

    """

    _MODEL_FILE_PATH = 'regression/linear_regression.pkl'
    _PARAM_NAMES = ['alpha', 'beta']

    def __init__(self, params: StanLinearRegressionParams):
        self._params = params

        if params.algorithm is None:
            self._algorithm = 'NUTS'
        else:
            self._algorithm = params.algorithm

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
            'shrinkage': self._params.shrinkage,
            'sigma_upper': self._params.sigma_upper
        }

        if self._stan_model is None:
            raise ValueError('stan model is not loaded.')
        self._stanfit = self._stan_model.sampling(data=data,
                                                  iter=self._params.n_itr,
                                                  chains=self._params.chains,
                                                  n_jobs=self._params.n_jobs,
                                                  warmup=self._params.warmup,
                                                  algorithm=self._algorithm,
                                                  verbose=self._params.verbose)
        return self

    def get_model_file_path(self):
        """

        Returns
        -------
        str
            the name of the pickled model file.
        """
        return self.__class__._MODEL_FILE_PATH

    def calc_distribution(self, X, params: dict):
        alpha = params['alpha']
        beta = params['beta']
        return X.dot(alpha.T) + beta


class StanLogisticRegression(BaseLinearRegressionModel, StanClassifierMixin,
                             StanModelLoadMixin):
    """
    Logistic regression using Stan.

    """

    _MODEL_FILE_PATH = 'regression/logistic_regression.pkl'
    _PARAM_NAMES = ['alpha', 'beta']

    def __init__(self,
                 chains: int,
                 warmup: int,
                 shrinkage: int,
                 n_jobs: int,
                 n_itr: int,
                 algorithm: str = 'NUTS',
                 verbose: bool = False):
        # TODO: use parameter class.
        self._stan_model = self.load_model()

    def _validate_params(self):
        pass

    def get_model_file_path(self):
        """

        Returns
        -------
        str
            the name of the pickled model file.
        """
        return self.__class__._MODEL_FILE_PATH

    def calc_distribution(self, X, params: dict):
        pass


class StanPoissonRegression(BaseLinearRegressionModel, StanClassifierMixin,
                            StanModelLoadMixin):
    """
    Poisson regression using Stan.
    """

    _MODEL_FILE_PATH = 'regression/poisson_regression.pkl'
    _PARAM_NAMES = ['alpha', 'beta']

    def __init__(self,
                 chains: int,
                 warmup: int,
                 shrinkage: int,
                 n_jobs: int,
                 n_itr: int,
                 algorithm: str = 'NUTS',
                 verbose: bool = False):
        # TODO: use parameter class.
        self._stan_model = self.load_model()

    def _validate_params(self):
        pass

    def get_model_file_path(self):
        """

        Returns
        -------
        str
            the name of the pickled model file.
        """
        return self.__class__._MODEL_FILE_PATH

    def calc_distribution(self, X, params: dict):
        pass
