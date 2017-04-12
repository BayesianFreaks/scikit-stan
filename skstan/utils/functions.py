import numpy as np


def sigmoid_each(x: np.array):
    return 1 / (1 + np.exp(-x))
