# Demo

```python
import numpy as np

from skstan.regression.linear_models import LinearRegression

if __name__ == '__main__':
    x = np.array([[1, 2, 5, 3, 2], [6, 5, 1, 1, 1]])
    y = np.array([1, 0])

    glm = LinearRegression(shrinkage=10, chains=8)
    stanfit = glm.fit(x, y)
    print(stanfit)
```

and we got result as followings.

```
Inference for Stan model: anon_model_d11135a2987c87996e739e8bb6d82895.
8 chains, each with iter=2000; warmup=1000; thin=1; 
post-warmup draws per chain=1000, total post-warmup draws=8000.

           mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
alpha[0]-6.2e-3     0.2   6.26 -12.02  -4.33-7.7e-3    4.2  12.06    999   1.01
alpha[1]  -0.05    0.28   8.06 -15.68  -5.43 2.5e-3   5.48  16.09    829   1.01
alpha[2]   0.09    0.17   5.89 -11.26  -3.98 5.2e-3   4.09   11.7   1149    1.0
alpha[3]    0.4    0.26   8.96 -17.14  -5.42   0.15   6.34  18.34   1165    1.0
alpha[4]  -0.34    0.25   9.66  -18.9   -6.8   -0.3   6.29  18.83   1451   1.01
beta       0.12    0.24   9.92 -19.62  -6.49   0.06   7.08  19.59   1651    1.0
sigma      0.32  4.8e-3   0.11   0.12   0.23   0.32   0.41   0.49    515   1.03
yp[0]       1.0  4.1e-3   0.34    0.3    0.8    1.0   1.19   1.71   6742    1.0
yp[1]    1.4e-4  3.9e-3   0.34  -0.71   -0.2 3.6e-4    0.2   0.72   7816    1.0
lp__      -1.87    0.05   1.84  -6.14  -2.85  -1.57  -0.54   0.82   1544   1.01

Samples were drawn using NUTS at Wed Apr 12 11:53:59 2017.
For each parameter, n_eff is a crude measure of effective sample size,
and Rhat is the potential scale reduction factor on split chains (at 
convergence, Rhat=1).
```


# What is scikit-stan
`scikit-stan` will enable you to use various bayesian models based on 
`stan`(http://mc-stan.org) and `pystan` with a elegant interface like a 
`scikit-learn`.

# How to use
## Install
```sh
git clone https://github.com/BayesianFreaks/scikit-stan
cd scikit-stan
python3 setup.py install
```

## Uninstall
```sh
pip3 uninstall scikit-stan
```

# Using python2?
Are you joking? 

We can't touch you because we are living in the future from you, and you're living in past ages. Please say hello to Nobunaga Oda.


We will always use newest features of the latest version of python, so you should use the latest version of python.
