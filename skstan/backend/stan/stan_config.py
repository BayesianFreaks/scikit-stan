

class StanBackendConfig:

    PKL_BASE_DIR = 'stan_model'

    MODEL_PKL_MAP = {
        'linear_regression': 'linear_regression.pkl',
        'logistic_regression': 'logistic_regression.pkl',
        'poisson_regression': 'poisson_regression.pkl'
    }
