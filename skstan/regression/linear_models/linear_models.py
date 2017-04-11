from ..base import RegressionModelMixin
from ..base import RegressionStanData


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
    def preprocess(dat: RegressionStanData) -> RegressionStanData:
        return dat.append(sigma_upper=dat['y'].std())


class LogisticRegression(RegressionModelMixin):
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


class PoissonRegression(RegressionModelMixin):
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

            yp <- exp(x*alpha + beta);
        }
        model{
            alpha ~ normal(0, shrinkage);
            beta ~ normal(0, shrinkage);

            y ~ poisson(yp);
        }
    '''
