from abc import ABCMeta
from abc import abstractmethod

import numpy as np


class LinearModel(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, x: np.array, y: np.array):
        pass
