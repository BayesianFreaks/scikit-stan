import os
import pickle

import numpy as np

from skstan import PROJECT_ROOT


class StanModelLoadMixin:
    """
    Stan model loader class.
    This class provides the methods to load a compiled stan model.

    """

    def load_model(self):
        """
        Load a pickled stan model specified by the name which
        `get_model_file_name` method return and return `StanModel` object.
        Parameters
        ----------

        Returns
        -------
        StanModel
            A StanModel object specified by model_name argument.
        """

        model_file_name = self.get_model_file_path()
        pkl_base_dir = self._base_dir()
        pkl_file_path = os.path.join(pkl_base_dir, model_file_name)
        with open(pkl_file_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _base_dir():
        return os.path.join(PROJECT_ROOT, 'stan_model')


class StanClassifierMixin:

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

        return np.apply_along_axis(np.median, 1, self.predict_dist(X))

    def predict_dist(self, X):
        """
        Parameters
        ----------
        X: array-like


        Returns
        -------
        array, shape (n_samples,)
            distributions.

        """
        if self._stanfit is None:
            raise ValueError("model is not trained.")

        param_dict = {}
        for param_name in self._PARAMS:
            param = self._stanfit.extract()[param_name]
            param_dict[param_name] = param

        return self.calc_distribution(param_dict)
