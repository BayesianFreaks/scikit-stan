from abc import ABCMeta, abstractclassmethod

import numpy as np

from skstan.pystan import LinearRegressionPystanMixin
from skstan.pystan import LinearRegressionStanParamMixin
from skstan.pystan import StanFit


class LinearRegressionBase(metaclass=ABCMeta):

    def __init__(self, shrinkage: float):
        self.__validate_params(shrinkage)
        self.shrinkage = shrinkage

    @staticmethod
    def __validate_params(shrinkage: float):
        if 0.0 >= shrinkage:
            raise ValueError('shrinkage parameter must be positive')

    @property
    @abstractclassmethod
    def model_code(self):
        pass

    @abstractclassmethod
    def fit(self, x: np.array, y: np.array):
        pass

    @abstractclassmethod
    def inv_link(self, x: np.array):
        pass


class LinearRegressionMixin(LinearRegressionPystanMixin,
                            LinearRegressionStanParamMixin):

    def _default_fit(self, x: np.array, y: np.array, shrinkage: float):
        data = self.stan_data(x, y, shrinkage=shrinkage)
        return StanFit(self, self.inference(data=data))

