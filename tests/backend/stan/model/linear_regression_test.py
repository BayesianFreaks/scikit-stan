from skstan.backend.stan.model import StanLinearRegression
from skstan.backend.stan.model import StanLogisticRegression
from skstan.backend.stan.model import StanPoissonRegression
from skstan.backend.stan.model.base_model import StanModelLoadMixin
from skstan.params import StanLinearRegressionParams


class TestStanLinearRegression:

    def test_get_model_file_name(self, monkeypatch):
        # Test that model file name is returned.

        def mock_load_model(dummy):
            return None

        monkeypatch.setattr(StanModelLoadMixin,
                            'load_model', mock_load_model)
        params = StanLinearRegressionParams(
            chains=3, warmup=1000, n_itr=5000, n_jobs=1, algorithm='NUTS',
            verbose=False, shrinkage=10,
        )
        slr = StanLinearRegression(params)
        actual = slr.get_model_file_path()
        expected = 'regression/linear_regression.pkl'
        assert actual == expected


class TestStanLogisticRegression:

    def test_get_model_file_name(self, monkeypatch):
        # Test that model file name is returned.
        def mock_load_model(dummy):
            return None

        monkeypatch.setattr(StanModelLoadMixin,
                            'load_model', mock_load_model)
        slr = StanLogisticRegression(chains=3, warmup=1000, shrinkage=10,
                                     n_jobs=1, n_itr=5000)
        actual = slr.get_model_file_path()
        expected = 'regression/logistic_regression.pkl'
        assert actual == expected


class TestStanPoissionRegression:

    def test_get_model_file_name(self, monkeypatch):
        # Test that model file name is returned.
        def mock_load_model(dummy):
            return None

        monkeypatch.setattr(StanModelLoadMixin,
                            'load_model', mock_load_model)
        spr = StanPoissonRegression(chains=3, warmup=1000, shrinkage=10,
                                    n_jobs=1, n_itr=5000)
        actual = spr.get_model_file_path()
        expected = 'regression/poisson_regression.pkl'
        assert actual == expected
