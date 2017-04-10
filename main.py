import skstan.utils.functions
from skstan.linear_models.linear_models import LinearRegression
import numpy as np

if __name__ == '__main__':

    x = np.array([[1,2,3], [6,5,4]])
    y = np.array([6,5])

    rgr = LinearRegression()

    stanfit = rgr.fit(x, y)

    print(stanfit)
