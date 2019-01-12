import json
import os

TMP_DIR = '/tmp'

SKSTAN_HOME_ENV_VAR = 'SKSTAN_HOME'
SKSTAN_DIR = '.skstan'
SKSTAN_JSON = 'skstan.json'

DEFAULT_BACKGROUND = 'stan'


def extract_skstan_config(skstan_dir):
    config_file_path = os.path.join(skstan_dir, SKSTAN_JSON)
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            return json.load(f)


# if SKSTAN_HOME environment variable exists, use its value as skstan dir.
if SKSTAN_HOME_ENV_VAR in os.environ:
    _skstan_dir = os.environ.get(SKSTAN_HOME_ENV_VAR)
else:
    _base_dir = os.path.expanduser('~')
    if os.access(_base_dir, os.W_OK):
        _skstan_dir = os.path.join(_base_dir, SKSTAN_DIR)
    else:
        _skstan_dir = TMP_DIR

try:
    _config_json = extract_skstan_config(_skstan_dir)
    _CURRENT_BACKEND = _config_json['backend']
except (KeyError, TypeError):
    # TODO: fix error handling when skstan json does not exist.
    _CURRENT_BACKEND = DEFAULT_BACKGROUND
