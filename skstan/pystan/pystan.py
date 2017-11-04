import numpy as np
import pystan as ps


class StanFit:

    def __init__(self, fit):
        self.fit = fit

    def __str__(self):
        return self.fit.__str__()


class PystanMixin:

    def inference(self, data, **kwargs):
        return ps.stan(model_code=self.model_code, data=data, **kwargs)


class LinearRegressionPystanMixin(PystanMixin):

    def distribution(self, x: np.array, stanfit: StanFit):
        if stanfit is None:
            raise ValueError('stanfit is not initialized. call fit function before prediction of distribution.')
        a = lambda: stanfit.fit.extract()['alpha']
        b = lambda: stanfit.fit.extract()['beta']
        return self.inv_link(x.dot(a().T) + b())
