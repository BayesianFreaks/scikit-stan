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

We can't see you because we are living in the future from you, and you're living in past ages. Please say hallo to Nobunaga Oda.

We will always use newest features of newest versions of python, and you should use newest version of python.
