import os
import pickle

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

        model_file_name = self.get_model_file_name()
        pkl_base_dir = self._base_dir()
        pkl_file_path = os.path.join(pkl_base_dir, model_file_name)
        with open(pkl_file_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _base_dir():
        return os.path.join(PROJECT_ROOT, 'stan_model')
