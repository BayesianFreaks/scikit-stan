import sys

from setuptools import find_packages
from setuptools import setup
from skstan import __version__

sys.path.append('./skstan')
sys.path.append('./test')


def rst_readme():
    try:
        from pypandoc import convert
        readme_text = convert('README.md', 'rst')
        return readme_text
    except ImportError:
        print("warning: pypandoc module not found, could not convert Markdown to RST")
        with open('README.md') as f:
            return f.read()


description = "Various bayesian models based on stan and pystan with a elegant interface like a scikit-learn or keras."

setup(
    name='skstan',
    version=__version__,
    url='https://skstan.org/latest/doc/',
    packages=find_packages(exclude=['test*']),
    description=description,
    long_description=rst_readme(),
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
