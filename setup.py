from setuptools import find_packages, setup

setup(
    name='spectral-multifidelity',
    packages=find_packages(include=['specmf']),
    version='0.1.0',
    description='Library to perform spectral multifidelity modeling',
    author='Orazio Pinti, orazio.pinti@gmail.com',
    install_requires=['numpy', 'scipy', 'matplotlib', 'scikit-learn'],
)
