from abc import ABCMeta, abstractstaticmethod
from typing import Dict

import numpy as np


class BaseStanData(Dict):
    def append(self, **kwargs):
        data = self.copy()
        data.update(kwargs)
        return data


class BaseModel(metaclass=ABCMeta):
    model_code: str

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractstaticmethod
    def preprocess(dat: BaseStanData) -> BaseStanData:
        pass


class BaseModelResult(metaclass=ABCMeta):

    @abstractstaticmethod
    def predict(self, x: np.array) -> np.array:
        pass

    @abstractstaticmethod
    def predict_dist(self, x: np.array) -> np.array:
        pass
