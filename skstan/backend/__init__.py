import json
import os
import sys

from skstan.backend.base_backend import BaseBackend

__all__ = [
    'BaseBackend',
]

TMP_DIR = '/tmp'

SKSTAN_HOME_ENV_VAR = 'SKSTAN_HOME'
SKSTAN_DIR = '.skstan'
SKSTAN_JSON = 'skstan.json'

DEFAULT_BACKGROUND = 'stan'


def _read_skstan_json(skstan_dir: str):
    config_file_path = os.path.join(skstan_dir, SKSTAN_JSON)
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            return json.load(f)


def _save_skstan_json(skstan_dir: str, current_backend: str):
    if not os.path.exists(skstan_dir):
        try:
            os.makedirs(skstan_dir)
        except OSError:
            pass

    config_file_path = os.path.join(skstan_dir, SKSTAN_JSON)
    if not os.path.exists(config_file_path):
        config_dict = {
            'backend': current_backend
        }
        try:
            with open(config_file_path, 'w') as f:
                f.write(json.dumps(config_dict, indent=2))
        except IOError:
            pass


# if $SKSTAN_HOME environment variable exists,
# use its value as skstan directory.
if SKSTAN_HOME_ENV_VAR in os.environ:
    _skstan_dir = os.environ.get(SKSTAN_HOME_ENV_VAR)
else:
    _home_dir = os.path.expanduser('~')
    if os.access(_home_dir, os.W_OK):
        _skstan_dir = os.path.join(_home_dir, SKSTAN_DIR)
    else:
        _skstan_dir = TMP_DIR

try:
    _config_dict = _read_skstan_json(_skstan_dir)
    _CURRENT_BACKEND = _config_dict['backend']
except (KeyError, TypeError):
    # TODO: fix error handling when skstan json does not exist.
    _CURRENT_BACKEND = DEFAULT_BACKGROUND

# try to save skstan config json.
_save_skstan_json(_skstan_dir, _CURRENT_BACKEND)

# Import the current Backend classes.
if _CURRENT_BACKEND == 'stan':
    sys.stderr.write('Using Stan backend.\n')
    from .stan import Backend
elif _CURRENT_BACKEND == 'tfp':
    sys.stderr.write('Using TensorFlow Probability backend.\n')
    from .tfp import Backend
else:
    raise ValueError(
        'Failed to import a backend class. Backend: {}'.format(_CURRENT_BACKEND))


def get_current_backend():
    """
    Return the name of the current backend as string.

    Returns
        the name of the backend (stan or tfp).
    -------

    """
    return _CURRENT_BACKEND
