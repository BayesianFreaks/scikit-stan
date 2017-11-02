import numpy as np

from skstan.pystan import StanFit
from skstan.regression import LinearRegressionBase
from skstan.regression import LinearRegressionMixin
from skstan.utils import sigmoid_each


class LinearRegression(LinearRegressionBase,
                       LinearRegressionMixin):

    def __init__(self, shrinkage: float, **kwargs):
        self._kwargs = kwargs
        super().__init__(shrinkage)

    def fit(self, x: np.array, y: np.array):
        data = self.stan_data(x, y, shrinkage=self.shrinkage, has_std=True)
        stan_fit = StanFit(
            self,
            self.inference(data=data, **self._kwargs)
        )
        return stan_fit

    def inv_link(self, x: np.array) -> np.array:
        return x

    model_code = """
        data{
            int n;
            int f;
            matrix[n,f] x;
            vector[n] y;
            real shrinkage;
            real sigma_upper;

        }
        parameters{
            vector[f] alpha;
            real beta;
            real<lower=0> sigma;
        }
        transformed parameters{
            vector[n] yp;

            yp <- x*alpha + beta;
        }
        model{
            alpha ~ normal(0, shrinkage);
            beta ~ normal(0, shrinkage);
            sigma ~ uniform(0, sigma_upper);

            y ~ normal(yp, sigma);
        }
    """


class LogisticRegression(LinearRegressionBase,
                         LinearRegressionMixin):

    def fit(self, x: np.array, y: np.array):
        return self._default_fit(x, y, shrinkage=self.shrinkage)

    def inv_link(self, x: np.array) -> np.array:
        return sigmoid_each(x)

    model_code = """
        data{
            int n;
            int f;
            matrix[n,f] x;
            int y[n];
            real shrinkage;

        }
        parameters{
            vector[f] alpha;
            real beta;
        }
        transformed parameters{
            vector[n] yp;

            yp <- x*alpha + beta;
        }
        model{
            alpha ~ normal(0, shrinkage);
            beta ~ normal(0, shrinkage);

            y ~ bernoulli_logit(yp);
        }
    """


class PoissonRegression(LinearRegressionBase,
                        LinearRegressionMixin):

    def fit(self, x: np.array, y: np.array):
        return self._default_fit(x, y, shrinkage=self.shrinkage)

    def inv_link(self, x: np.array) -> np.array:
        return np.exp(x)

    model_code = """
        data{
            int n;
            int f;
            matrix[n,f] x;
            int y[n];
            real shrinkage;

        }
        parameters{
            vector[f] alpha;
            real beta;
        }
        transformed parameters{
            vector[n] yp;

            yp <- x*alpha + beta;
        }
        model{
            alpha ~ normal(0, shrinkage);
            beta ~ normal(0, shrinkage);

            y ~ poisson(exp(yp));
        }
    """
