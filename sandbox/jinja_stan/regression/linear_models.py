import numpy as np

from sandbox.jinja_stan.stan import StanCodeMixin
from sandbox.jinja_stan.base import LinearModel


class LinearRegression(LinearModel, StanCodeMixin):

    def __init__(self):
        pass

    def _generate_stan_code(self):
        # TODO: add concrete implementations.
        data = []
        params = []
        transformed_params = []
        model = []

        stan_code = self.generate_stan_code(
            data,
            params,
            transformed_params,
            model
        )
        return stan_code

    def fit(self, x: np.array, y: np.array):
        # TODO: implement fit method.
        pass
