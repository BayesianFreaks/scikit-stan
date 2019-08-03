from typing import Sequence

from skstan.backend import Backend
from skstan.backend import get_current_backend
from skstan.model.lgm import BaseLinearRegression
from skstan.params import StanLinearRegressionParams
from skstan.params import TFPLinearRegressionParams


class LinearRegression(BaseLinearRegression):
    """
    Linear regression model class.

    Parameters
    ----------
    chains: int
        The number of chains.
    warmup: int
        Markov chains may take some time before the chain settles into the
        equilibrium distribution. The samples generated during the initial
        phase are discarded. The warmup value is the number of steps to be
        discarded.
    n_itr: int
        The number of iteration.
    n_jobs: int (default=-1)
        The number of jobs to run in parallel.
    algorithm: str (default=None)
       the algorithm name to be used in Stan.
    shrinkage: int (default=10)
        The standard deviation for non-information prior distribution.
        Usually, the deviation value is very large.

    """

    def __init__(self,
                 chains: int,
                 warmup: int,
                 n_itr: int,
                 n_jobs: int = -1,
                 algorithm: str = None,
                 verbose: bool = False,
                 shrinkage: float = 10.0):

        self._validate_params(chains, shrinkage)
        params = self._pack_model_params(
            chains=chains,
            warmup=warmup,
            n_itr=n_itr,
            n_jobs=n_jobs,
            algorithm=algorithm,
            verbose=verbose,
            shrinkage=shrinkage
        )
        self._backend_model = self._create_backend_model(params)

    @staticmethod
    def _create_backend_model(params):
        # TODO: create backend model in constructor.
        return Backend.LinearRegression(params)

    @staticmethod
    def _pack_model_params(chains: int, warmup: int, n_itr: int, n_jobs: int,
                           algorithm: str, shrinkage: float, verbose: bool):
        current_backend = get_current_backend()
        if current_backend == 'stan':
            return StanLinearRegressionParams(
                chains=chains,
                warmup=warmup,
                n_itr=n_itr,
                n_jobs=n_jobs,
                algorithm=algorithm,
                verbose=verbose,
                shrinkage=shrinkage
            )
        elif current_backend == 'tfp':
            # TODO: set parameters.
            return TFPLinearRegressionParams(verbose=verbose)
        else:
            raise ValueError('unknown backend: {}'.format(current_backend))

    @staticmethod
    def _validate_params(chains: int, shrinkage: float):
        if chains < 0:
            raise ValueError('chains must be positive.')
        if shrinkage < 0.0:
            raise ValueError('shrinkage must be positive.')

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]):
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

    def predict(self, X: Sequence[Sequence[float]]) -> Sequence[float]:
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

    def predict_proba(self, X: Sequence[Sequence[float]]) -> Sequence[float]:
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
        pass
        # self._backend_model = Backend.LogisticRegression()

    def _validate_params(self):
        pass

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]):
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

    def predict(self, X: Sequence[Sequence[float]]) -> Sequence[float]:
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
    """
    Poisson regression model class.

    Parameters
    ----------

    """

    def __init__(self):
        pass
        # self._backend_model = Backend.PoissonRegression()

    def _validate_params(self):
        pass

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]):
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

    def predict(self, X: Sequence[Sequence[float]]) -> Sequence[float]:
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
