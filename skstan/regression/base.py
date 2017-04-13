import pystan as ps

from ..base import BaseModel
from ..base import BaseStanData

from typing import Sequence, TypeVar

T = TypeVar('T')


class RegressionStanData(BaseStanData):
    def __init__(self, x: Sequence[T], y: Sequence[T], shrinkage: float):
        super().__init__()
        assert len(y.shape) == 1, 'Mismatch dimension. y must be 1 dimensional array'
        assert len(x.shape) == 2, 'Mismatch dimension. x must be 2 dimensional array'
        assert y.shape[0] == x.shape[0], 'Mismatch dimension. x and y must have same number of rows'

        self.data = {
            'x': x,
            'y': y,
            'n': x.shape[0],
            'f': x.shape[1],
            'shrinkage': shrinkage,
        }


class RegressionModel(BaseModel):
    def __init__(self, shrinkage: float, **kwargs):
        super().__init__(**kwargs)
        assert 0 < shrinkage, 'shrinkage parameter must be positive'

        self.shrinkage = shrinkage

    def fit(self, x: Sequence[T], y: Sequence[T]):
        return ps.stan(
            model_code=self.model_code,
            data=self.preprocess(
                RegressionStanData(x, y, self.shrinkage)
            ).data,
            **self.kwargs
        )

    @staticmethod
    def preprocess(dat: RegressionStanData) -> RegressionStanData:
        return dat
