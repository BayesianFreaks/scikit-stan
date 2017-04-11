import numpy as np
import pystan as ps

from ..base import BaseModel
from ..base import BaseStanData


class RegressionStanData(BaseStanData):
    def __init__(self, x: np.array, y: np.array, shrinkage: float):
        super().__init__()
        try:
            assert len(y.shape) == 1

        except AssertionError:
            raise ValueError(
                'Mismatch dimension. y must be 1 dimensional array'
            )

        try:
            assert len(x.shape) == 2

        except AssertionError:
            raise ValueError(
                'Mismatch dimension. x must be 2 dimensional array'
            )

        try:
            assert y.shape[0] == x.shape[0]

        except AssertionError:
            raise ValueError(
                'Mismatch dimension. x and y must have same number of rows'
            )

        self.data = {
            'x': x,
            'y': y,
            'n': x.shape[0],
            'f': x.shape[1],
            'shrinkage': shrinkage,
        }


class RegressionModelMixin(BaseModel):
    def __init__(self, shrinkage: float, **kwargs):
        super().__init__(**kwargs)
        try:
            assert 0 < shrinkage

        except AssertionError:
            raise ValueError(
                'shrinkage parameter must be positive'
            )

        self.shrinkage = shrinkage

    def fit(self, x: np.array, y: np.array):
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
