# scikit-stan

[![Build Status](https://travis-ci.org/BayesianFreaks/scikit-stan.svg?branch=master)](https://travis-ci.org/BayesianFreaks/scikit-stan) [![codecov](https://codecov.io/gh/BayesianFreaks/scikit-stan/branch/master/graph/badge.svg)](https://codecov.io/gh/BayesianFreaks/scikit-stan)
[![MIT license](https://img.shields.io/badge/License-MIT-green.svg)](https://lbesson.mit-license.org/)
[![](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)

## What is scikit-stan
`scikit-stan` will enable you to use various bayesian models based on 
`stan`(http://mc-stan.org) and `pystan` with an elegant interface like a
`scikit-learn` or `keras`.

## Demo

```python
import numpy as np

from skstan.regression.linear_models import LogisticRegression

if __name__ == '__main__':
    x = np.array(
        [
            [1,2,3,],
            [1,2,7,],
            [1,0,3,],
            [1,1,3,],
            [3,7,3,],
        ]
    )
    y = np.array([0,0,0,0,1])

    glm = LogisticRegression(shrinkage=10, chains=8)
    fit = glm.fit(x, y)
```

Then we got result object `fit`, and field `stanfit` is a stanfit object of pystan.

```python
print(fit.stanfit)
```

It gives following

```
Inference for Stan model: anon_model_f63cd5ccdd67c22034b2490ae4c9cdd1.
4 chains, each with iter=2000; warmup=1000; thin=1; 
post-warmup draws per chain=1000, total post-warmup draws=4000.

           mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
alpha[0]   2.23    0.29   8.88 -14.39  -3.87   2.02   8.07  20.67    966    1.0
alpha[1]   7.81    0.18   5.29  -1.08   4.01   7.48  11.23  19.01    880    1.0
alpha[2]  -9.79    0.22   5.87 -22.91 -13.41  -9.37  -5.38  -0.17    728    1.0
beta      -2.48    0.29    9.8 -22.63  -9.03   -2.3   3.99  16.91   1146    1.0
yp[0]    -13.99    0.32  11.19 -40.69 -20.24 -11.35  -5.42    0.3   1259    1.0
yp[1]    -53.15    1.14  32.08 -128.4 -71.99 -48.46 -29.35  -5.24    790    1.0
yp[2]    -29.61     0.6  16.66 -67.24 -39.44 -27.97  -17.0  -4.37    771    1.0
yp[3]     -21.8    0.44  13.17 -52.98 -29.03 -19.57 -11.91  -3.23    894    1.0
yp[4]     29.51    0.69  24.68   0.58   10.3  23.36  42.72  90.17   1276    1.0
lp__      -2.16    0.05   1.48  -5.93   -2.9  -1.81  -1.07  -0.32    956    1.0

Samples were drawn using NUTS at Thu Apr 13 07:52:33 2017.
For each parameter, n_eff is a crude measure of effective sample size,
and Rhat is the potential scale reduction factor on split chains (at 
convergence, Rhat=1).
```

Result object of `skstan` also have prediction methods. 
Predicted values can be obtained as samples from distribution with a `predict_dist` method, because it is bayesian model.

```python
yp_dist = fit.predict_dist(x)
print(yp_dist)
```
Then we got

```
array([[  2.63886682e-08,   5.23976746e-04,   5.54863097e-05, ...,
          2.46008578e-08,   3.74830192e-01,   3.45994043e-03],
       [  1.07746578e-22,   1.01664809e-18,   4.12813154e-26, ...,
          5.64992544e-19,   7.24386097e-12,   1.75795155e-23],
       [  8.04688037e-22,   4.44522113e-12,   1.42920488e-11, ...,
          7.71565191e-13,   5.13118658e-05,   4.26331280e-05],
       [  4.60810657e-15,   4.82743551e-08,   2.81612678e-08, ...,
          1.37772153e-10,   5.51614998e-03,   3.84594197e-04],
       [  9.99999998e-01,   1.00000000e+00,   1.00000000e+00, ...,
          9.99965378e-01,   1.00000000e+00,   1.00000000e+00]])
```

So let's check the histgram of first row with `pandas.Series`.

```python
import pandas as pd
pd.Series(yp_dist[0]).hist(bins=20)
```
![Histgram of first row](https://dfh6g0ekreo71.cloudfront.net/images/chart/hist.png) <!-- change this url to a appropriate one after setting domain. -->


If you need a median of samples, you can get it with just `predict` method


```python
yp = fit.predict(x)
print(yp)
```
gives
```
array([  1.17280235e-05,   9.01419773e-22,   7.16023732e-13,
         3.18368664e-09,   1.00000000e+00])
```


## How to install
### Install

Installers for the latest released version are available at [PyPI](https://pypi.python.org/pypi/skstan).

```sh
pip3 install skstan
```

### Install from sources

```sh
git clone https://github.com/BayesianFreaks/scikit-stan
cd scikit-stan
python3 setup.py install
```

### Uninstall
```sh
pip3 uninstall scikit-stan
```

# Using python2?
Are you joking? 

We can't touch you because we are living in the future from you, and you're living in past ages. Please say hello to Nobunaga Oda.

We will always use newest features of the latest version of python, so you should use the latest version of python.


# Models

## Ready
### Regression Models
- Linear Regrassion
- Poisson Regression
- Logistic Regression

## Next Steps
### Regression Models
- Gamma Regression
- GLMM
- etc...

### Time Series Models
- AR Model
- MA Model
- ARMA Model
- ARIMA Model
- ARCH Model
- GARCH Model
- TAR Model
- State Space Model
- or Some Dynamic Regression Models
- etc...

### Clustering Model
- Gaussian Mixture Model
- Latent Dirichlet Allocation
- etc...

### Particular Application
- Modeling about online-advertisement
- Decompose time series data
- Empirical Bayesian Estimation
