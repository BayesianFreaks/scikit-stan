import numpy as np
import pystan as ps

from ..base import BaseModel, BaseStanData


class RegressionStanData(BaseStanData):
    def __init__(self, x: np.array, y: np.array, shrinkage: float):
        super().__init__()
        assert len(y.shape) == 1
        assert len(x.shape) == 2
        assert shrinkage >= 0

        self.data = {
            'x': x,
            'y': y,
            'n': x.shape[0],
            'f': x.shape[1],
            'shrinkage': shrinkage,
        }


class RegressionModelMixin(BaseModel):
    def __init__(self, shrinkage: float):
        self.shrinkage = shrinkage

    def fit(self, x: np.array, y: np.array):
        return ps.stan(
            model_code=self.model_code,
            data=self.__preprocess(
                RegressionStanData(x, y, self.shrinkage)
            ).data
        )

    @staticmethod
    def __preprocess(dat: RegressionStanData) -> RegressionStanData:
        return dat


class LinearRegression(RegressionModelMixin):
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
    def __preprocess(dat: RegressionStanData) -> RegressionStanData:
        return dat.append(sigma_upper=dat['y'].std())
