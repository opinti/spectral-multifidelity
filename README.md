# Graph-Laplacian based Bayesian Spectral Multifidelity Library

## Overview

A Python package for graph-laplacian based multi-fidelity modeling.

![Graphical Abstract](figures/graphical-abstract.png)

## Features

- Perform multi-fidelity modeling using Bayesian specmf method
- Simple `Graph` class to store graphs and their representations
- Method computations happen in a dedicated `MultiFidelityModel` class  
- Utilities for preprocessing and visualize data for multi-fidelity modeling
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

The code has been tested with Python 3.12.

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
from specmf.plot import *


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
    'kappa': 2,
    'method': 'full',
}
model = MultiFidelityModel(**model_config)

# Transform the data
mf_data, mf_covar_mat, mf_var = model.transform(
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

![Example-1](figures/example-1.png)


### 3. Using the clustering functionalities to select high-fidelity data

The `MultiFidelityModel` class also supports spectral clustering operations. 
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

![Example Clustering](figures/example-clustering.png)


Clustering can be used to find the nodes where to acquire high-fidelity data:

```python
# Aquire high-fidelity data at the centroids location
hf_data = lf_data[inds_centroids, :] + 2

# Transform the data
mf_data, mf_covar_mat, mf_var = model.transform(
    graph_lf,
    hf_data,
    inds_centroids,
)
model.summary()


# Plot the datasets
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(lf_data[:, 0], lf_data[:, 1], color='orange', label='LF data')
ax.scatter(lf_data[inds_centroids, 0], lf_data[inds_centroids, 1], color='blue', label='HF data')
ax.scatter(mf_data[:, 0], mf_data[:, 1], color='green', label='MF data')
ax.scatter(hf_data[:, 0], hf_data[:, 1], color='red')
for i in range(len(inds_centroids)):
    ax.plot(
        [lf_data[inds_centroids[i], 0], hf_data[i, 0]],
        [lf_data[inds_centroids[i], 1], hf_data[i, 1]],
        color='grey',
        linewidth=1.,
        alpha=0.75,
)
ax.legend()
ax.set_xlabel(r"$u_1$", fontsize=16)
ax.set_ylabel(r"$u_2$", fontsize=16, rotation=0, labelpad=20)
```

![Example-2](figures/example-2.png)


You also have uncertainty accosiated to each multi-fidelity esimate, which can be visualized in this simple 2-d example:

```python
# Plot the variance of multi-fidelity esimates
fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
scatter = ax.scatter(mf_data[:, 0], mf_data[:, 1], c=mf_var, s=50, vmin=0.025)
ax.scatter(mf_data[inds_centroids, 0], mf_data[inds_centroids, 1], c='red', s=50, label='Training data')
ax.set_title("Variance of multi-fidelity estimates", fontsize=16)
ax.set_xlabel(r"$u_1$", fontsize=16)
ax.set_ylabel(r"$u_2$", fontsize=16, rotation=0, labelpad=20)
fig.colorbar(scatter, ax=ax,)
ax.legend(loc="lower right", fontsize=14)
```

![Example-variance](figures/example-variance.png)


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

## Reference

Source code of the paper: 

**Graph Laplacian-based Bayesian Multi-fidelity Modelinge**  
O. Pinti, J. M. Budd, F. Hoffmann, A. A. Oberai.
[arXiv:2409.08211](https://arxiv.org/abs/2409.08211)
