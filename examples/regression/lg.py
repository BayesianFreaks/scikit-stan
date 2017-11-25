import numpy as np

from skstan.regression.linear_models.linear_model_v2 import LinearRegression


def liner_regression():
    x = np.array([[1, 2, 5, 3, 2], [6, 5, 1, 1, 1]])
    y = np.array([1, 0])

    glm = LinearRegression(shrinkage=10, chains=8)

    model = glm.fit(x, y)
    print('inference')
    print(model.stanfit)

    yp_dist = model.predict_dist(x)
    print('distribution')
    print(yp_dist)


if __name__ == '__main__':
    liner_regression()
