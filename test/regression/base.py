from unittest import TestCase

import numpy as np

from skstan.regression.base import RegressionModelMixin
from skstan.regression.base import RegressionStanData


class TestRegressionModelMixin(TestCase):
    def test_preprocess(self):
        dat1 = RegressionStanData(
            np.array([[1, 1, ], [2, 2, ]]),
            np.array([3, 3]),
            10
        )
        self.assertEqual(dat1, RegressionModelMixin.preprocess(dat1))
        