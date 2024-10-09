# Bayesian Spectral Multifidelity Library

author: Orazio Pinti, PhD
email:  orazio.pinti@gmail.com

## Overview

A Python package for graph-laplacian based multi-fidelity modeling. 

## Features

- Performs multi-fidelity modeling using Bayesian specmf method
- Defines simple Graph classe to store graphs and their representations
- The method computations have a dedicated MultiFidelityModel class  
- Utilities for preprocessing data for multi-fidelity modeling
- Includes several experiments for testing the models

## Installation

### Using `pip`

You can install the package and its dependencies by running:

```bash
pip install .
```

For development mode:

```bash
pip install -e .
```

### Dependencies

The project requires the following libraries:

- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`

Code has been tested with Python 3.12 or above.

## Usage

### Importing the package

Once installed, you can import the main module and start using the classes and functions provided by the package.

```python
from specmf import MultiFidelityModel, Graph
```

### Example Usage

For examples on how to use please check the experiment notebooks.

## Project Structure

```bash
spectral-multifidelity/
│
├── data/                  # Contains datasets for experiments
├── specmf/                # Main code
│   ├── __init__.py        # Initialization file
│   ├── models.py          # Contains model and graph classes
│   ├── preprocess.py      # Data preprocessing utilities
│   ├── graph_core.py      # Core graph class
│   ├── utils.py           # Utility functions
├── notebooks/             # Notebooks with experiments
├── tests/                 # Unit tests
├── setup.py               # Setup file for packaging
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation (this file)
```

## Contact

For any questions or issues, please reach out at orazio.pinti@gmail.com 

## Citations

If you use this package in your research, please cite the following paper:

**Graph Laplacian-based Bayesian Multi-fidelity Modelinge**  
O. Pinti, J. M. Budd, F. Hoffmann, A. A. Oberai.
[arXiv:2409.08211](https://arxiv.org/abs/2409.08211)
