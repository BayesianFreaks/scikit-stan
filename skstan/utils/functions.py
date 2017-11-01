import numpy as np


def sigmoid_each(x: np.array):
    return 1.0 / (1.0 + np.exp(-x))
