from unittest import TestCase

import numpy as np

from skstan.utils.functions import sigmoid_each


class TestFunctions(TestCase):

    def test_sigmoid_each(self):
        """
        Check sigmoid_each calculates np.array values of sigmoid function.
        """
        x = np.array([1.0, 2.0])
        actual = sigmoid_each(x)

        expected = np.array([0.73105858, 0.88079708])
        self.assertTrue(np.allclose(actual, expected))
