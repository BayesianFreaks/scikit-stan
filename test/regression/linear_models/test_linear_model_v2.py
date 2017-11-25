from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from skstan.regression.linear_models.linear_model_v2 import LinearRegression
from skstan.regression.linear_models.linear_model_v2 import LogisticRegression


class TestLinearRegression(TestCase):

    def test_fit(self):
        x = np.array([[1, 2, 5, 3, 2], [6, 5, 1, 1, 1]])
        y = np.array([1, 0])
        model = LinearRegression(shrinkage=10, chains=8)
        model.stan_data = Mock(return_value={
            'x': x, 'y': y, 'n': x.shape[0], 'f': x.shape[1],
            'shrinkage': 10, 'sigma_upper': y.std()})
        model.inference = Mock(return_value='stanfit_object')
        model.fit(x, y)

        stan_data_call_args = model.stan_data.call_args[0]
        stan_data_call_args_with_name = model.stan_data.call_args[1]

        # check stan_data method is called with correct arguments.
        np.testing.assert_array_equal(stan_data_call_args[0], x)
        np.testing.assert_array_equal(stan_data_call_args[1], y)
        self.assertEqual(stan_data_call_args_with_name['shrinkage'], 10)
        self.assertEqual(stan_data_call_args_with_name['additional_params'], {'sigma_upper': 0.5})

        actual_data = model.inference.call_args[1]

        # check inference method is called with correct arguments.
        np.testing.assert_array_equal(actual_data['data']['x'], x)
        np.testing.assert_array_equal(actual_data['data']['y'], y)
        self.assertEqual(actual_data['data']['n'], 2)
        self.assertEqual(actual_data['data']['f'], 5)
        self.assertEqual(actual_data['data']['shrinkage'], 10)
        self.assertEqual(actual_data['data']['sigma_upper'], 0.5)
        self.assertEqual(actual_data['chains'], 8)

    def test_predict_dict(self):
        model = LinearRegression(shrinkage=10, chains=8)
        returned_dist = np.array([1.0, 2.0])
        model.distribution = Mock(return_value=returned_dist)
        model.stanfit = 'stanfit_object'
        x = np.array([[1, 2, 5, 3, 2], [6, 5, 1, 1, 1]])
        result = model.predict_dist(x)

        # check distribution method is called with correct arguments.
        distribution_call_args = model.distribution.call_args[0]
        np.testing.assert_array_equal(distribution_call_args[0], x)
        self.assertEqual(distribution_call_args[1], 'stanfit_object')

        # check predict_dict method returns the expected dist.
        np.testing.assert_array_equal(result, returned_dist)

    def test_inv_link(self):
        model = LinearRegression(shrinkage=10, chains=8)
        x = np.array([[1, 2, 5, 3, 2], [6, 5, 1, 1, 1]])
        actual = model.inv_link(x)

        # ink_inv(inverse link function) method calculates the value of linear function.
        np.testing.assert_array_equal(actual, x)


class TestLogisticRegression(TestCase):

    def test_fit(self):
        x = np.array([[], []])
        y = np.array([])
        model = LogisticRegression(shrinkage=10)

    def test_predict_dict(self):
        pass

    def test_inv_link(self):
        pass
