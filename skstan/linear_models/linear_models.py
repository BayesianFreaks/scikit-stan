from abc import ABCMeta, abstractmethod
import numpy as np
import pystan as ps

class RegressionModelMixin():
    model_code: str

    

    def fit(self, x: np.array, y: np.array):
        assert len(y.shape) == 1
        assert len(x.shape) == 2
        data = {
            'x': x,
            'y': y,
            'n': x.shape[0],
            'f': x.shape[1],
        }

        return ps.stan(model_code=self.model_code, data=data)


class LinearRegression(RegressionModelMixin):
    model_code = '''
        data{
            int n;
            int f;
            matrix[n,f] x;
            vector[n] y;
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
            alpha ~ normal(0, 100);
            beta ~ normal(0, 100);
            sigma ~ uniform(0, 10);

            y ~ normal(yp, sigma);
        }
    '''

