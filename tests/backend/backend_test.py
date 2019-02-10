import importlib
import json
import os
import shutil

from assertpy import assert_that

import skstan.backend
from skstan.backend.stan.stan_backend import StanBackend
from skstan.backend.tfp.tfp_backend import TFPBackend


def setup_function():
    if os.path.exists('/tmp/.skstan'):
        shutil.rmtree('/tmp/.skstan')


def teardown_function():
    if os.path.exists('/tmp/.skstan'):
        shutil.rmtree('/tmp/.skstan')


def test_import_backend_without_config(monkeypatch):
    # Test that the default backend is used
    # if a config json does not exist.
    def mock_tmp_directory(dummy):
        return '/tmp'

    # The .skstan directory which contains skstan.json is located at the local
    # home directory by default. Using monkeypatch, the value of the variable
    # which represents the location of .skstan directory will be replaced to
    # /tmp.
    monkeypatch.setattr(os.path, 'expanduser', mock_tmp_directory)
    importlib.reload(skstan.backend)

    with open('/tmp/.skstan/skstan.json', 'r') as f:
        config_dict = json.load(f)

    # check that `/tmp/.skstan` directory is created.
    assert_that(os.path.exists('/tmp/.skstan')).is_true()
    assert 'stan' == config_dict['backend']
    assert skstan.backend._CURRENT_BACKEND == 'stan'

    current_backend = skstan.backend.Backend()
    assert_that(current_backend).is_instance_of(StanBackend)


def test_import_backend_with_config(monkeypatch):
    # Test that the existing config json is read.
    def mock_tmp_directory(dummy):
        return '/tmp'

    test_config = {
        'backend': 'tfp'
    }

    # Create .skstan directory and write config json before testing.
    os.makedirs('/tmp/.skstan')
    with open('/tmp/.skstan/skstan.json', 'w') as f:
        f.write(json.dumps(test_config, indent=2))

    # The .skstan directory which contains skstan.json is located at the local
    # home directory by default. Using monkeypatch, the value of the variable
    # which represents the location of .skstan directory will be replaced to
    # /tmp.
    monkeypatch.setattr(os.path, 'expanduser', mock_tmp_directory)
    importlib.reload(skstan.backend)

    assert skstan.backend._CURRENT_BACKEND == 'tfp'

    current_backend = skstan.backend.Backend()
    assert_that(current_backend).is_instance_of(TFPBackend)
