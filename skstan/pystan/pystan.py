import numpy as np
import pystan as ps


class PystanMixin:

    def inference(self, data: dict, **kwargs):
        return ps.stan(self.model_code(), data, **kwargs)


class PystanLinearRegressionMixin(PystanMixin):

    def distribution(self, x: np.array):
        a = self.stanfit.extract()['alpha']
        b = self.stanfit.extract()['beta']

        return self.inv_link(x.dot(a().T) + b())
