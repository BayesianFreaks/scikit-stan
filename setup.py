import sys

from setuptools import setup, find_packages

sys.path.append('./skstan')
sys.path.append('./test')


def readme():
    try:
        with open('README.md') as f:
            readme = f.read()
            return readme
    except IOError:
        return ''


setup(
    name='skstan',
    version='0.0',
    packages=find_packages(exclude=['test*']),
    description='',
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
        'pyyaml'
    ],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    license="MIT"
)
