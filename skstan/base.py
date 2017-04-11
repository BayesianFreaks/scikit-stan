from abc import ABCMeta, abstractstaticmethod
from collections import UserDict


class BaseStanData(UserDict):
    def append(self, **kwargs):
        self.data.update(kwargs)
        return BaseStanData(
            self.data
        )


class BaseModel(metaclass=ABCMeta):
    model_code: str

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractstaticmethod
    def preprocess(dat: BaseStanData) -> BaseStanData:
        pass
