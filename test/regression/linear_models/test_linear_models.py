from unittest import TestCase

import numpy as np

from skstan.regression.base import RegressionStanData
from skstan.regression.linear_models import LinearRegression


class TestLinearRegression(TestCase):
    def setUp(self):
        self.dat = {
            'x': np.ndarray(shape=(2, 2)),
            'y': np.array([0, 0]),
            'shrinkage': 10
        }

    def test_preprocess(self):
        self.assertAlmostEqual(
            LinearRegression.preprocess(
                RegressionStanData(**self.dat)
            )['sigma_upper'],
            0
        )
