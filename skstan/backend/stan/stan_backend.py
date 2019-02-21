import os
import pickle

from skstan import PROJECT_ROOT
from skstan.backend.stan import StanLinearRegression
from skstan.backend.stan import StanLogisticRegression
from skstan.backend.stan import StanPoissonRegression


class StanBackend:
    """
    Stan backend class.
    This class provides model clases which are implemented by using Stan.

    """

    LinearRegression = StanLinearRegression
    LogisticRegression = StanLogisticRegression
    PoissionRegression = StanPoissonRegression

    def __init__(self):
        # print backend name to check .
        print('stan backend.')


class StanModelLoader:
    """
    Stan model loader class.

    This class provides the static method (class method) to load a compiled
    stan model.

    """

    _PKL_BASE_DIR = os.path.join(PROJECT_ROOT, 'stan_model')

    # _MODEL_PKL_MAP = {
    #     LINEAR_REGRESSION: 'linear_regression.pkl',
    #     LOGISTIC_REGRESSION: 'logistic_regression.pkl',
    #     POISSON_REGRESSION: 'poisson_regression.pkl'
    # }

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
