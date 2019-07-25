from typing import Sequence

import numpy as np

from skstan.model.lgm import LinearRegression


class TestLinearRegression:
    def test_fit(self, mocker):
        # Test that logistic regression is trained by using fit method.

        class DumyBackendModel:
            def __init__(self,
                         chains: int,
                         warmup: int,
                         shrinkage: int,
                         n_jobs: int,
                         n_itr: int,
                         algorithm: str = 'NUTS',
                         verbose: bool = False):
                pass

            def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]):
                print('called dummy model')

        dummy_backend = DumyBackendModel(
            chains=3,
            warmup=1000,
            shrinkage=10,
            n_jobs=1,
            n_itr=5000,
            algorithm='NUTS')

        lr = LinearRegression(
            chains=3, warmup=1000, n_jobs=1, n_itr=5000, algorithm='NUTS')

        # replace backend model to dummy one.
        # monkeypatch.setattr(lr, '_backend_model', dummy_backend)
        mocker.patch.obejct('linear_regression.Backend')

        X = np.array([[5.1, 3.5, 1.4, 0.2], [5.4, 3.4, 1.7, 0.2],
                      [7., 3.2, 4.7, 1.4], [5., 2., 3.5, 1.],
                      [5.9, 3.2, 4.8, 1.8], [6.3, 3.3, 6., 2.5],
                      [6.9, 3.2, 5.7, 2.3]])

        y = np.array([
            0,
            0,
            1,
            1,
            1,
            2,
            2,
        ])

        lr.fit(X, y)
