# Graph-Laplacian based Bayesian Spectral Multifidelity Library

author: Orazio Pinti

email:  orazio.pinti@gmail.com

## Overview

A Python package for graph-laplacian based multi-fidelity modeling.

![Graphical Abstract](graphical-abstract/graphical-abstract.png)

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
- `scikit-optimize`
- `umap-learn`
- `matplotlib`

Code has been tested with Python 3.12 or above.

## Usage

### Importing the package

Once installed, you can import the main module and start using the classes and functions provided by the package.

```python
from specmf import MultiFidelityModel, Graph
```

### Example Usage

For end-to-end examples on how to use, please check the experiment notebooks.

The main module of the package is `models`. Below are examples demonstrating how to use these modules.

### 1. Using the `Graph` class from `models.py`

The `Graph` class inherhits from `GraphCore` class, which computes graph-based representations, such as the adjacency matrix and the normalized graph Laplacian. 
Here is an example of how to initialize the class and compute an adjacency matrix.

```python
import numpy as np
from specmf.models import Graph


# Create some sample data
data = np.random.rand(10, 3)  # 10 samples with 3 features each

# Initialize GraphCore with default parameters
graph = Graph(data=data)

# Compute adjacency matrix
adjacency = graph.adjacency

# Compute normalized graph Laplacian
graph_laplacian = graph.graph_laplacian

print(f"{adjacency.shape=}")
print(f"{graph_laplacian.shape=}")
```


### 2. Using the `MultiFidelityModel` class from `models.py`
The `MultiFidelityModel` class is designed to perform multi-fidelity modeling. 
It allows you to transform all nodes of a graph based on a few more accurate nodes data.

```python
import numpy as np
from specmf.models import Graph, MultiFidelityModel


# Create some sample data
np.random.seed(42)
n_lf = 100
n_hf = 10
lf_data = np.random.rand(n_lf, 2)
hf_inds = np.random.choice(n_lf, n_hf) # Assume we aquire high-fi data at these indices
hf_data = lf_data[hf_inds, :] + 2

# Create the graph
graph_config = {
    'metric': 'euclidean',
    'dist_space': 'ambient',
    'method': 'full',
    'corr_scale': None,
    'k_adj': 7,
    'p': 0.5,
    'q': 0.5,
}
graph_lf = Graph(
    data=lf_data,
    **graph_config,
)

# Initialize the model
model_config = {
    'sigma': 0.01,
    'kappa': 0.1,
    'method': 'full',
}
model = MultiFidelityModel(**model_config)

# Transform the data
mf_data, mf_var, dPhi = model.transform(
    graph_lf,
    hf_data,
    hf_inds,
)
model.summary()

# Plot the datasets
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(lf_data[:, 0], lf_data[:, 1], color='orange', label='LF data')
ax.scatter(lf_data[hf_inds, 0], lf_data[hf_inds, 1], color='blue', label='HF data')
ax.scatter(mf_data[:, 0], mf_data[:, 1], color='green', label='MF data')
ax.scatter(hf_data[:, 0], hf_data[:, 1], color='red')
for i in range(len(hf_inds)):
    ax.plot(
        [lf_data[hf_inds[i], 0], hf_data[i, 0]],
        [lf_data[hf_inds[i], 1], hf_data[i, 1]],
        color='grey',
        linewidth=1.,
        alpha=0.75,
)
ax.legend()
ax.set_xlabel(r"$u_1$", fontsize=16)
ax.set_ylabel(r"$u_2$", fontsize=16, rotation=0, labelpad=20)
```

### 3. Using the clustering functionalities to select high-fidelity data

The `MultiFidelityModel` class also supports clustering operations. 
Here's an example of clustering using the cluster method.

```python
# Perform clustering on the graph to find the high-fidelity indices
inds_centroids, labels = model.cluster(graph_lf, n=n_hf)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(lf_data[:, 0], lf_data[:, 1], c=labels, cmap='tab10')

ax.legend()
ax.set_xlabel(r"$u_1$", fontsize=16)
ax.set_ylabel(r"$u_2$", fontsize=16, rotation=0, labelpad=20)
```

This can be used to find the nodes where to acquire high-fidelity data. Based on the example above:

```python
hf_data = lf_data[inds_centroids, :] + 2

# Transform the data
mf_data, mf_var, dPhi = model.transform(
    graph_lf,
    hf_data,
    hf_inds,
)
model.summary()


# Plot the datasets
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(lf_data[:, 0], lf_data[:, 1], color='orange', label='LF data')
ax.scatter(lf_data[hf_inds, 0], lf_data[hf_inds, 1], color='blue', label='HF data')
ax.scatter(mf_data[:, 0], mf_data[:, 1], color='green', label='MF data')
ax.scatter(hf_data[:, 0], hf_data[:, 1], color='red')
for i in range(len(hf_inds)):
    ax.plot(
        [lf_data[hf_inds[i], 0], hf_data[i, 0]],
        [lf_data[hf_inds[i], 1], hf_data[i, 1]],
        color='grey',
        linewidth=1.,
        alpha=0.75,
)
ax.legend()
ax.set_xlabel(r"$u_1$", fontsize=16)
ax.set_ylabel(r"$u_2$", fontsize=16, rotation=0, labelpad=20)
```

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
