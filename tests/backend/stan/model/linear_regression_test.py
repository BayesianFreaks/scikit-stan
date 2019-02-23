from skstan.backend.stan.model import StanLinearRegression
from skstan.backend.stan.model import StanLogisticRegression
from skstan.backend.stan.model import StanPoissonRegression
from skstan.backend.stan.model.base_model import BaseStanModel


class TestStanLinearRegression:

    def test_get_model_file_name(self, monkeypatch):
        # Test that model file name is returned.

        def mock_load_model(dummy):
            return None

        monkeypatch.setattr(BaseStanModel,
                            'load_model', mock_load_model)
        slr = StanLinearRegression()
        actual = slr.get_model_file_name()
        expected = 'linear_regression.pkl'
        assert actual == expected


class TestStanLogisticRegression:

    def test_get_model_file_name(self, monkeypatch):
        # Test that model file name is returned.
        def mock_load_model(dummy):
            return None

        monkeypatch.setattr(BaseStanModel,
                            'load_model', mock_load_model)
        slr = StanLogisticRegression()
        actual = slr.get_model_file_name()
        expected = 'logistic_regression.pkl'
        assert actual == expected


class TestStanPoissionRegression:

    def test_get_model_file_name(self, monkeypatch):
        # Test that model file name is returned.
        def mock_load_model(dummy):
            return None

        monkeypatch.setattr(BaseStanModel,
                            'load_model', mock_load_model)
        spr = StanPoissonRegression()
        actual = spr.get_model_file_name()
        expected = 'poisson_regression.pkl'
        assert actual == expected
