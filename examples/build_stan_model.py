import os
import pickle

from pystan import StanModel


def build_stan_model(stan_file_path):
    """
    Build Stan model for example codes.
    """
    total_stan_file_path = '../stan/' + stan_file_path
    pkl_file_name = os.path.join(
        '../skstan/stan_model', os.path.dirname(stan_file_path),
        os.path.basename(stan_file_path.replace('.stan', '.pkl')))

    if not os.path.exists(pkl_file_name):
        stan_model = StanModel(file=total_stan_file_path, verbose=True)
        dir_name = os.path.dirname(pkl_file_name)

        # Make a directory if `skstan/stan_model/:model_name` directory
        # does not exist.
        if not os.path.exists(dir_name):
            print('target directory does not exist.')
            print(dir_name)
            os.makedirs(dir_name)
        # Output picke file. The file is not tracked by git.
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(stan_model, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    stan_file_list = [
        'regression/linear_regression.stan',
        'regression/logistic_regression.stan'
    ]

    for file_path in stan_file_list:
        build_stan_model(file_path)
