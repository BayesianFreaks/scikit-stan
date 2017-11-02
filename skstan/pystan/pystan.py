import numpy as np
import pystan as ps


class PystanMixin:

    def inference(self, data, **kwargs):
        return ps.stan(model_code=self.model_code, data=data, **kwargs)


class LinearRegressionPystanMixin(PystanMixin):

    def distribution(self, x: np.array):
        a = self._stanfit.extract()['alpha']
        b = self._stanfit.extract()['beta']

        return self.inv_link(x.dot(a().T) + b())


class StanFit:

    def __init__(self, model, fit):
        self.model = model
        self.fit = fit

    def __str__(self):
        return self.fit.__str__()
