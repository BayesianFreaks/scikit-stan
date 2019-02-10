import os
import pickle

from pystan import StanModel

from skstan import PROJECT_ROOT
from skstan.backend import BaseBackend
from skstan.model.lgm import (
    LINEAR_REGRESSION,
    LOGISTIC_REGRESSION,
    POISSON_REGRESSION
)


class StanBackend(BaseBackend):
    """
    Stan backend class.

    Wrapper of `PyStan`.
    """

    def __init__(self):
        # print backend name to check .
        print('stan backend.')

    def sample(self):
        pass


class StanModelLoader:
    """
    Stan model loader class.

    This class provides the static method (class method) to load a compiled
    stan model.
    """

    _PKL_BASE_DIR = os.path.join(PROJECT_ROOT, 'stan_model')

    _MODEL_PKL_MAP = {
        LINEAR_REGRESSION: 'linear_regression.pkl',
        LOGISTIC_REGRESSION: 'logistic_regression.pkl',
        POISSON_REGRESSION: 'poisson_regression.pkl'
    }

    @classmethod
    def load_stan_model(cls, model_name: str):
        """
        Load a pickled stan model and return a `StanModel` instance specified
        by the argument.
        `StanModel` class is belong to PyStan.

        Parameters
        ----------
        model_name: str
            A name of model to load.

        Returns
        -------
        StanModel
            A StanModel object specified by model_name argument.
        """
        pkl_file_name = cls._MODEL_PKL_MAP[model_name]
        pkl_file_path = os.path.join(cls._PKL_BASE_DIR, pkl_file_name)
        with open(pkl_file_path, 'rb') as f:
            return pickle.load(f)
