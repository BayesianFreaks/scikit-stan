import os
import pickle

from skstan import PROJECT_ROOT
from skstan.utils.field import (
    LINEAR_REGRESSION,
    LOGISTIC_REGRESSION,
    POISSON_REGRESSION
)


class StanBackend:
    """
    Stan backend class.

    Wrapper of `PyStan`.
    """

    def __init__(self):
        pass

    def sample(self):
        pass


class StanModelLoader:
    """
    Stan model loader class.

    This class provides the static method to load a compiled stan models.
    """

    PKL_BASE_DIR = os.path.join(PROJECT_ROOT, 'stan_model')

    MODEL_PKL_MAP = {
        LINEAR_REGRESSION: 'linear_regression.pkl',
        LOGISTIC_REGRESSION: 'logistic_regression.pkl',
        POISSON_REGRESSION: 'poisson_regression.pkl'
    }

    @staticmethod
    def load_stan_model(model_name):
        """
        Load a pickled stan model and return a `StanModel` instance specified
        by the argument. `StanModel` class is belong to PyStan.

        Parameters
        ----------
        model_name: str
            A name of model to load.

        Returns
        -------
        StanModel
            A StanModel object specified by model_name argument.
        """
        pkl_file_name = StanModelLoader.MODEL_PKL_MAP[model_name]
        pkl_file_path = os.path.join(StanModelLoader.PKL_BASE_DIR,
                                     pkl_file_name)
        with open(pkl_file_path, 'rb') as f:
            return pickle.load(f)
