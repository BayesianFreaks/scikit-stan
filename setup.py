import glob
import os
import pickle
import sys
from distutils.command.build_py import build_py

from setuptools import find_packages
from setuptools import setup

import skstan

sys.path.append('./skstan')
sys.path.append('./tests')

PKL_STAN_MODEL_BASE_DIR = 'skstan/stan_model'
STAN_CODE_DIR = 'stan'

VERSION = skstan.__version__


def rst_readme():
    try:
        from pypandoc import convert
        readme_text = convert('README.md', 'rst')
        return readme_text
    except ImportError:
        print("warning: pypandoc module not found, could not convert Markdown to RST")
        with open('README.md') as f:
            return f.read()


def build_and_output_stan_model(target_base_dir):
    from pystan import StanModel

    def is_stan_file(file_name):
        _, ext = os.path.splitext(file_name)
        return ext == '.stan'

    stan_file_list = [name for name in
                      glob.glob(STAN_CODE_DIR + '/**', recursive=True)
                      if is_stan_file(name)]

    for stan_file in stan_file_list:
        with open(stan_file) as f:
            stan_code = f.read()

        stan_model = StanModel(model_code=stan_code)
        pkl_file_name = os.path.basename(stan_file.replace('.stan', '.pkl'))
        target_sub_dir_name = os.path.dirname(stan_file).replace('stan/', '')
        target_dir_name = os.path.join(target_base_dir, target_sub_dir_name)
        target_file_name = os.path.join(target_dir_name, pkl_file_name)

        # output pickle file to a path `stan_model/{model_group}/{pickle_file}`
        if not os.path.exists(target_dir_name):
            os.mkdir(target_dir_name)
        with open(target_file_name, 'wb') as f:
            pickle.dump(stan_model, f, protocol=pickle.HIGHEST_PROTOCOL)


class BuildCmd(build_py):
    """
    build stan models at setup.
    """

    def run(self):
        if not self._dry_run:
            # build stan model and output pickled file.
            target_dir = os.path.join(self.build_lib, PKL_STAN_MODEL_BASE_DIR)
            self.mkpath(target_dir)
            build_and_output_stan_model(target_dir)
        build_py.run(self)


DESCRIPTION = """
scikit-stan is a high-level Bayesian analysis API written in Python.
"""

INSTALL_REQUIRES = [
    'pystan'
]

setup(
    name='skstan',
    version=VERSION,
    url='https://skstan.org',
    packages=find_packages(exclude=['tests*']),
    description=DESCRIPTION,
    long_description=rst_readme(),
    author='scikit-stan developers',
    author_email='scikit-stan@googlegroups.com',
    test_suite='tests',
    package_data={
        '': ['*.yaml']
    },
    install_requires=INSTALL_REQUIRES,
    setup_requires=[
        'pystan',
    ],
    extras_require={
        'tests': ['pytest']
    },
    cmdclass={
        'build_py': BuildCmd,
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    zip_safe=False,
    license="MIT"
)
