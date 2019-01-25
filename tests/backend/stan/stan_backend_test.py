from pystan import StanModel

from skstan.backend.stan import StanModelLoader
from tests import TEST_ROOT


class TestStanModelLoader:

    def test_load_stan_model(self):
        # Test that a pickled stan model is loaded and deserialized.
        StanModelLoader.PKL_BASE_DIR = TEST_ROOT + '/test_model'
        model_name = 'linear_regression'
        actual_model = StanModelLoader.load_stan_model(model_name)

        assert StanModel == type(actual_model)
