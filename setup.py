import sys

from setuptools import find_packages
from setuptools import setup

sys.path.append('./skstan')
sys.path.append('./test')


def rst_readme():
    try:
        from pypandoc import convert_file
        readme_text = convert_file('README.md', 'rst')
        return readme_text.decode('utf-8')
    except ImportError:
        print("warning: pypandoc module not found, could not convert Markdown to RST")
        with open('README.md', 'r') as f:
            return f.read()


def description():
    desc = "Various bayesian models based on stan and pystan with a elegant interface like a scikit-learn or keras."
    return desc


setup(
    name='skstan',
    version='0.0.0_6',
    url='https://skstan.org/latest/doc/',
    packages=find_packages(exclude=['test*']),
    description=description(),
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
