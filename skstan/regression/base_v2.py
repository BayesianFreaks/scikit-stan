from abc import ABCMeta, abstractclassmethod

import numpy as np


class LinearRegressionBase(metaclass=ABCMeta):

    @abstractclassmethod
    def fit(self, x: np.array, y: np.array):
        pass

    @abstractclassmethod
    def preprocess(self, data: dict):
        pass

    @abstractclassmethod
    def inv_link(self, x: np.array):
        pass
