import numpy as np


class StanParamMixin:
    pass


class LinearRegressionStanParamMixin(StanParamMixin):

    def stan_data(self, x: np.array, y: np.array, shrinkage: float, has_std=False):
        self.__validate_params(x, y)
        data = {
            'x': x,
            'y': y,
            'n': x.shape[0],
            'f': x.shape[1],
            'shrinkage': shrinkage,
        }
        if not has_std:
            return data
        else:
            data.update({'sigma_upper': y.std()})
            return data

    @staticmethod
    def __validate_params(x, y):
        if len(y.shape) != 1:
            raise ValueError('Mismatch dimension. y must be 1 dimensional array')
        if len(x.shape) != 2:
            raise ValueError('Mismatch dimension. x must be 2 dimensional array')
        if y.shape[0] != x.shape[0]:
            raise ValueError('Mismatch dimension. x and y must have same number of rows')
