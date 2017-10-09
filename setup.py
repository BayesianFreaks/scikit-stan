import sys

from setuptools import find_packages
from setuptools import setup

sys.path.append('./skstan')
sys.path.append('./test')


def rst_readme():
    with open('README.md') as f:
        readme_text = f.read()
    try:
        from pypandoc import convert_text
        return convert_text(readme_text, 'rst', format='md').replace("\\r\\n", "\\n")
    except ImportError:
        print("warning: pypandoc module not found, could not convert Markdown to RST")
        return readme_text


def description():
    desc = "Various bayesian models based on stan and pystan with a elegant interface like a scikit-learn or keras."
    return desc


setup(
    name='skstan',
    version='0.0.0_4',
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
