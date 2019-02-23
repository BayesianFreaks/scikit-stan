from abc import ABCMeta
from abc import abstractmethod

from skstan.backend.stan.model.loader import StanModelLoader


class BaseStanModel(metaclass=ABCMeta):

    @abstractmethod
    def get_model_file_name(self):
        pass

    def load_model(self):
        """
        Returns
        -------
        StanModel
            A StanModel object.
        """

        loader = StanModelLoader()
        model_file_name = self.get_model_file_name()
        return loader.load_stan_model(model_file_name)
