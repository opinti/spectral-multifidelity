from setuptools import find_packages, setup

setup(
    name="spectral-multifidelity",
    packages=find_packages(include=["specmf"]),
    version="0.1.0",
    description="Library to perform spectral multifidelity modeling",
    author="Orazio Pinti, orazio.pinti@gmail.com",
    install_requires=[
        "numpy==1.26.0",
        "matplotlib==3.8.0",
        "scipy==1.11.3",
        "scikit-learn==1.5.1",
        "pytest==8.3.3",
        "scikit-optimize==0.8.1",
        "umap-learn>=0.5.6",
    ],
)
