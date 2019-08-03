import numpy as np

from skstan.model.lgm.linear_regression import LinearRegression


class TestLinearRegression:
    def test_fit(self, mocker, monkeypatch):
        # Test that logistic regression is trained by using fit method.

        MockBackendModel = mocker.Mock()
        mock_model = MockBackendModel()
        MockBackendModel.fit.return_value = None

        def dummy_create_backend_model(self, params):
            print('create backend')
            return mock_model

        # Replace backend class to mock one defined above.
        monkeypatch.setattr(LinearRegression, '_create_backend_model',
                            dummy_create_backend_model)

        lr = LinearRegression(
            chains=3, warmup=1000, n_jobs=1, n_itr=5000, algorithm='NUTS')

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

        mock_model.fit.assert_called_once_with(X, y)
