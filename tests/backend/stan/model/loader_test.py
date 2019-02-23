from pystan import StanModel

from skstan.backend.stan.model.loader import StanModelLoader
from tests import TEST_ROOT


class TestStanModelLoader:

    def test_load_stan_model(self, monkeypatch):
        # Test that a pickled stan model is loaded and deserialized.

        # Replace the base directory of pickled files to the test directory.
        monkeypatch.setattr(StanModelLoader, '_PKL_BASE_DIR',
                            TEST_ROOT + '/test_model')
        model_file_name = 'linear_regression.pkl'
        actual_model = StanModelLoader.load_stan_model(model_file_name)

        assert StanModel == type(actual_model)
