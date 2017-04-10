import numpy as np

from skstan.linear_models.linear_models import LinearRegression

if __name__ == '__main__':
    x = np.array([[1, 2, 3], [6, 5, 4]])
    y = np.array([6, 5])

    rgr = LinearRegression(shrinkage=10)

    stanfit = rgr.fit(x, y)

    print(stanfit)
