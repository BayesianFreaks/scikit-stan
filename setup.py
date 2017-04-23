import sys

from setuptools import setup, find_packages

sys.path.append('./skstan')
sys.path.append('./test')

setup(
    name='skstan',
    version='0.0',
    packages=find_packages(exclude=['test*']),
    test_suite='test',
    package_data={
        '': ['*.yaml']
    },
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'pystan',
        'pyyaml'
    ]
)
