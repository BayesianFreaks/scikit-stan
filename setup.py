import sys

from setuptools import find_packages
from setuptools import setup

sys.path.append('./skstan')
sys.path.append('./tests')


def rst_readme():
    try:
        from pypandoc import convert
        readme_text = convert('README.md', 'rst')
        return readme_text
    except ImportError:
        print("warning: pypandoc module not found, could not convert Markdown to RST")
        with open('README.md') as f:
            return f.read()


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
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    license="MIT"
)
