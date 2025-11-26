from setuptools import find_packages, setup


setup(
    name="spectral-multifidelity",
    version="0.1",
    description="Library to perform Bayesian spectral multi-fidelity modeling",
    author="Orazio Pinti",
    author_email="orazio.pinti@gmail.com",
    url="https://github.com/opinti/spectral-multifidelity",
    license="MIT",
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.12",
    install_requires=[
        "numpy==1.26.0",
        "matplotlib==3.8.0",
        "scipy==1.11.3",
        "scikit-learn==1.5.1",
        "pytest==8.3.3",
        "scikit-optimize==0.8.1",
        "umap-learn==0.5.6",
        "pyyaml>=5.1",
    ],
    extras_require={
        "dev": [
            "pytest",
            "ruff",
        ],
    },
)
