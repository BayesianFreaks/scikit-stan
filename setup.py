import sys

from setuptools import setup, find_packages

sys.path.append('./skstan')
sys.path.append('./test')


def readme():
    try:
        with open('README.rst') as f:
            readme = f.read()
            return readme
    except IOError:
        return ''


description = "Various bayesian models based on stan and pystan with a elegant interface like a scikit-learn or keras."

setup(
    name='skstan',
    version='0.0.0d',
    url='https://skstan.org/latest/doc/',
    packages=find_packages(exclude=['test*']),
    description=description,
    long_description=readme(),
    author='scikit-stan development team',
    author_email='scikit-stan@googlegroups.com',
    test_suite='test',
    package_data={
        '': ['*.yaml']
    },
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'pystan',
        'pyyaml',
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    license="MIT"
)
