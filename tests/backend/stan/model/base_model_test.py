from pystan import StanModel

from skstan.backend.stan.model.base_model import StanModelLoadMixin
from tests import TEST_ROOT


class TestStanModelLoadMixin:

    def test_load_model(self, monkeypatch):
        # Test that a pickled stan model is loaded and deserialized.

        class DummyModel(StanModelLoadMixin):
            def get_model_file_name(self):
                return 'linear_regression.pkl'

        def mock_base_dir(dummy):
            return TEST_ROOT + '/test_model'
        monkeypatch.setattr(DummyModel, '_base_dir', mock_base_dir)

        model = DummyModel()
        actual_model = model.load_model()

        assert StanModel == type(actual_model)
