import numpy as np
import pystan as ps

from ..base import BaseModel
from ..base import BaseModelResult
from ..base import BaseStanData


class RegressionStanData(BaseStanData):
    def __init__(self, x: np.array, y: np.array, shrinkage: float):
        super().__init__()
        assert len(y.shape) == 1, 'Mismatch dimension. y must be 1 dimensional array'
        assert len(x.shape) == 2, 'Mismatch dimension. x must be 2 dimensional array'
        assert y.shape[0] == x.shape[0], 'Mismatch dimension. x and y must have same number of rows'

        self.update(
            {
                'x': x,
                'y': y,
                'n': x.shape[0],
                'f': x.shape[1],
                'shrinkage': shrinkage,
            }
        )


class RegressionModel(BaseModel):
    def __init__(self, shrinkage: float, **kwargs):
        super().__init__(**kwargs)
        assert 0 < shrinkage, 'shrinkage parameter must be positive'

        self.shrinkage = shrinkage

    def fit(self, x: np.array, y: np.array):
        return RegressionModelResult(
            self,
            ps.stan(
                model_code=self.model_code,
                data=self.preprocess(
                    RegressionStanData(x, y, self.shrinkage)
                ),
                **self.kwargs
            )
        )

    @staticmethod
    def preprocess(dat: RegressionStanData) -> RegressionStanData:
        return dat

    @staticmethod
    def inv_link(x: np.array) -> np.array:
        return x


class RegressionModelResult(BaseModelResult):
    def __init__(self, model: RegressionModel, stanfit):
        self.model = model
        self.stanfit = stanfit

    def predict(self, x: np.array) -> np.array:
        return np.apply_along_axis(
            np.median,
            1,
            self.predict_dist(x)
        )

    def predict_dist(self, x: np.array) -> np.array:
        # lambda is used for lazy evaluation
        a = lambda: self.stanfit.extract()['alpha']
        b = lambda: self.stanfit.extract()['beta']

        return self.model.inv_link(x.dot(a().T) + b())
