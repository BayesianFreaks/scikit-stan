import numpy as np
import pystan as ps

from ..base import BaseModel
from ..base import BaseStanData


class RegressionStanData(BaseStanData):
    def __init__(self, x: np.array, y: np.array, shrinkage: float):
        super().__init__()
        assert len(y.shape) == 1
        assert len(x.shape) == 2
        assert y.shape[0] == x.shape[0]
        assert shrinkage >= 0

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
