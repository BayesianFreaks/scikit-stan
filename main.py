import numpy as np

from skstan.linear_models import PoissonRegression

if __name__ == '__main__':
    x = np.array([[1, 2], [6, 5]])
    y = np.array([1, 0])

    glm = PoissonRegression(shrinkage=10)

    stanfit = glm.fit(x, y)

    print(stanfit)
