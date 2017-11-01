import numpy as np
import pystan as ps

from skstan.pystan import PyStanMixin
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


class RegressionModelResult(BaseModelResult, PyStanMixin):

    def __init__(self, model_code: str, data, **kwargs):
        self.pystan = self._inference(model_code, data, **kwargs)

    @classmethod
    def model_result(cls, model_code, data, **kwargs):
        return RegressionModelResult(model_code, data, **kwargs)

    def stanfit(self):
        return self._stanfit()

    def predict(self, x: np.array) -> np.array:
        return np.apply_along_axis(
            np.median,
            1,
            self._predict_dist(x)
        )

    def _predict_dist(self, x: np.array) -> np.array:
        # lambda is used for lazy evaluation
        a = self.pystan.extract_func('alpha')
        b = self.pystan.extract_func('beta')
        return self.model.inv_link(x.dot(a().T) + b())
