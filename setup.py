import glob
import os
import pickle
import sys
from distutils.command.build_py import build_py

from pystan import StanModel
from setuptools import find_packages
from setuptools import setup

sys.path.append('./skstan')
sys.path.append('./tests')

PKL_STAN_MODEL_BASE_DIR = './stan_model'


def rst_readme():
    try:
        from pypandoc import convert
        readme_text = convert('README.md', 'rst')
        return readme_text
    except ImportError:
        print("warning: pypandoc module not found, could not convert Markdown to RST")
        with open('README.md') as f:
            return f.read()


def build_stan_model():
    def is_stan_file(file_name):
        _, ext = os.path.splitext(file_name)
        return ext == '.stan'

    stan_file_list = [name for name in glob.glob('stan/**', recursive=True) if is_stan_file(name)]
    for stan_file in stan_file_list:
        with open(stan_file) as f:
            stan_code = f.read()

        stan_model = StanModel(model_code=stan_code)
        pkl_file_name = os.path.basename(stan_file.replace('.stan', '.pkl'))
        target_dir_name = os.path.join(PKL_STAN_MODEL_BASE_DIR, os.path.dirname(stan_file).replace('stan/', ''))

        # output pickle file to stan_model/{model_group}/{pickle_file}
        if not os.path.exists(target_dir_name):
            os.mkdir(target_dir_name)
        with open(os.path.join(target_dir_name, pkl_file_name), 'wb') as f:
            pickle.dump(stan_model, f, protocol=pickle.HIGHEST_PROTOCOL)


class BuildCmd(build_py):
    """
    build stan models.
    """
    def run(self):
        if not self._dry_run:
            # TODO: build stan models.
            pass
        build_py.run(self)


DESCRIPTION = """
scikit-stan is a high-level Bayesian analysis API written in Python.
"""

INSTALL_REQUIREMENTS = [
    'pystan'
]

setup(
    name='skstan',
    version='0.0.0-dev',
    url='https://skstan.org/latest/doc/',
    packages=find_packages(exclude=['tests*']),
    description=DESCRIPTION,
    long_description=rst_readme(),
    author='scikit-stan development team',
    author_email='scikit-stan@googlegroups.com',
    test_suite='tests',
    package_data={
        '': ['*.yaml']
    },
    install_requires=INSTALL_REQUIREMENTS,
    extras_require={
        'tests': ['pytest']
    },
    cmdclass={
        'build_py': BuildCmd
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    license="MIT"
)
