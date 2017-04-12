import numpy as np

from ..base import RegressionModel
from ..base import RegressionStanData
from ...utils import functions as f


class LinearRegression(RegressionModel):
    model_code = '''
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
    '''

    @staticmethod
    def preprocess(dat: RegressionStanData) -> RegressionStanData:
        return dat.append(sigma_upper=dat['y'].std())


class LogisticRegression(RegressionModel):
    model_code = '''
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
    '''

    @staticmethod
    def inv_link(x: np.array) -> np.array:
        return f.sigmoid_each(x)


class PoissonRegression(RegressionModel):
    model_code = '''
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
    '''

    @staticmethod
    def inv_link(x: np.array) -> np.array:
        return np.exp(x)
