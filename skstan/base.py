from collections import UserDict

from abc import ABCMeta

class BaseStanData(UserDict):
    def append(self, **kwargs):
        self.data.update(kwargs)
        return BaseStanData(
            self.data
        )


class BaseModel(metaclass=ABCMeta):
    model_code: str
