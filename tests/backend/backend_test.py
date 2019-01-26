import importlib
import os
import shutil

from assertpy import assert_that

import skstan.backend
import json


def setup_function():
    if os.path.exists('/tmp/.skstan'):
        shutil.rmtree('/tmp/.skstan')


def teardown_function():
    if os.path.exists('/tmp/.skstan'):
        shutil.rmtree('/tmp/.skstan')


def test_import_backend_without_config(monkeypatch):
    # test that the default backend is used
    # if a config json does not exist.
    def mock_tmp_directory(dummy):
        return '/tmp'

    monkeypatch.setattr(os.path, 'expanduser', mock_tmp_directory)
    importlib.reload(skstan.backend)

    assert skstan.backend._skstan_dir == '/tmp/.skstan'

    with open('/tmp/.skstan/skstan.json', 'r') as f:
        config_dict = json.load(f)

    # check that `/tmp/.skstan` directory is created.
    assert_that(os.path.exists('/tmp/.skstan')).is_true()
    assert 'stan' == config_dict['backend']
    assert skstan.backend._CURRENT_BACKEND == 'stan'


def test_import_backend_with_config(monkeypatch):
    # test that the existing config json is read.
    def mock_tmp_directory(dummy):
        return '/tmp'

    test_config = {
        'backend': 'tfp'
    }

    os.makedirs('/tmp/.skstan')
    with open('/tmp/.skstan/skstan.json', 'w') as f:
        f.write(json.dumps(test_config, indent=2))

    monkeypatch.setattr(os.path, 'expanduser', mock_tmp_directory)
    importlib.reload(skstan.backend)

    assert skstan.backend._skstan_dir == '/tmp/.skstan'
    assert skstan.backend._CURRENT_BACKEND == 'tfp'
