from abc import ABCMeta, abstractstaticmethod
from typing import Dict


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
    pass
