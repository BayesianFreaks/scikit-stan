import pystan as ps


class Pystan:

    def __init__(self, stanfit):
        self.stanfit = stanfit

    def extract_func(self, param_name):
        return lambda: self.stanfit.extract()[param_name]


class PyStanMixin:

    def _inference(self, model_code: str, data, **kwargs):
        return Pystan(ps.stan(model_code=model_code, data=data, **kwargs))

    def _stanfit(self):
        return self.pystan.stanfit

