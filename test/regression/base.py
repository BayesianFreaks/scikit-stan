from unittest import TestCase

import numpy as np

from skstan.regression.base import RegressionModelMixin
from skstan.regression.base import RegressionStanData


class TestRegressionModelMixin(TestCase):
    def test_preprocess(self):
        dat1 = RegressionStanData(
            np.ndarray(shape=(2, 2)),
            np.ndarray(shape=(2,)),
            10
        )
        self.assertEqual(dat1, RegressionModelMixin.preprocess(dat1))
        self.assertRaises(
            ValueError,
            lambda: RegressionModelMixin(shrinkage=-1)
        )

class TestRegressionStanData(TestCase):
    def setUp(self):
        self.good = {
            'x': np.ndarray(shape=(2, 2)),
            'y': np.ndarray(shape=(2,)),
            'shrinkage': 10
        }

        self.res_good = {
            'x': np.ndarray(shape=(2, 2)),
            'y': np.ndarray(shape=(2,)),
            'n': 2,
            'f': 2,
            'shrinkage': 10
        }

        self.x_dim_not_2 = {
            'x': np.ndarray(shape=(2, 2, 2)),
            'y': np.ndarray(shape=(2,)),
            'shrinkage': 10
        }

        self.y_dim_not_1 = {
            'x': np.ndarray(shape=(2, 2)),
            'y': np.ndarray(shape=(2, 2)),
            'shrinkage': 10
        }

        self.row_number_mismatch = {
            'x': np.ndarray(shape=(2, 2)),
            'y': np.ndarray(shape=(3,)),
            'shrinkage': 10
        }



    def test_constructor(self):
        self.assertRaises(ValueError, lambda: RegressionStanData(**self.x_dim_not_2))
        self.assertRaises(ValueError, lambda: RegressionStanData(**self.y_dim_not_1))
        self.assertRaises(ValueError, lambda: RegressionStanData(**self.row_number_mismatch))

