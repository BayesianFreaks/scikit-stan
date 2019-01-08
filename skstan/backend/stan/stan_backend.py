import os
import pickle

from skstan.backend.stan import StanBackendConfig


class StanBackend:

    @staticmethod
    def get_stan_model(model_name):
        pkl_file_name = StanBackendConfig.MODEL_PKL_MAP[model_name]
        pkl_file_path = os.path.join(StanBackendConfig.PKL_BASE_DIR, pkl_file_name)
        with open(pkl_file_path, 'rb') as f:
            return pickle.load(f)
