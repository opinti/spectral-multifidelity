# Graph Laplacian-based Bayesian Multi-fidelity Modeling

## Overview

A python package for graph laplacian-based multi-fidelity modeling.

![Graphical Abstract](figures/graphical-abstract.png)

## Features

- Perform multi-fidelity modeling using Bayesian specmf method.
- Simple custom `Graph` class to store graphs representations and perform spectral clustering.
- Modeling computations happen in a dedicated `MultiFidelityModel` class.
- Utilities for preprocessing and visualize data for multi-fidelity modeling.
- Includes several experiments for testing the models.

## Installation

You can install the package via pip:

```bash
cd local/path/spectral-multifidelity
pip install .
```

For development mode:

```bash
cd local/path/spectral-multifidelity
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

The package was developed and tested with Python 3.12.

## Usage

The following sections describe different functionalities provided by the package.
For end-to-end examples, please check the experiment notebooks.

### Example Usage

The main module of the package is `models.py`. Below are examples demonstrating how to use this module.

### 1. Using the `Graph` class from `models.py`

The `Graph` class inherhits from `GraphCore` class, which computes graph-based representations, such as the adjacency matrix and the normalized graph Laplacian. 
Here is an example of how to initialize an instance of the class and compute standard matrix representations such as adjacency and graph Laplacian matrix.

```python
import numpy as np
from specmf.models import Graph

## Create some sample data
data = np.random.rand(10, 3)

# Initialize a graph with default parameters
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
It allows you to transform all nodes of a graph based on a few more accurate nodes data, i.e. the "high-fidelity" data.


#### 2.1 Generate some synthetic data

First, let's generate some 2-d data to use in an illustrative example.
We consider a simple standard dataset provided by `sklearn`.

```python
from sklearn import datasets
np.random.seed(42)


def affine_transform(data, apply_stretch=True, apply_shift=True):
    """
    Applies an affine transformation to input data.

    Parameters:
    - data (np.ndarray): Input data of shape (n_samples, 2).
    - apply_stretch (bool, optional): If True, applies a 2x stretch along y=x and a 0.8x stretch along y=-x.
    - apply_shift (bool, optional): If True, applies a shift of 0.25 along both x and y axes.

    Returns:
    - np.ndarray: Transformed data
    """
    new_data = data.copy()
    if apply_stretch:
        P = (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]])
        D = np.array([[2, 0], [0, 0.8]])
        M = P @ D @ P.T
        new_data = (M @ data.T).T
    if apply_shift:
        shift = np.array([0.25, 0.25])
        new_data += shift
    return new_data


## Define low- and high-fidelity datasets
# Generate base data once without noise
n_samples = 1000
base_data, _ = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.,)

# Add different levels of noise to create low-fidelity and high-fidelity data
# An affine transformation is also applied to the high-fidelity data.
noise_scale_lf = 0.015
noise_scale_hf = 0.01
lf_noise = np.random.normal(scale=noise_scale_lf, size=base_data.shape)
hf_noise = np.random.normal(scale=noise_scale_hf, size=base_data.shape)

lf_data = base_data + lf_noise
hf_data = base_data + hf_noise
hf_data = affine_transform(hf_data)
```

This is what the low- and high-fidelity datasets look like:

```python
# Plot the datasets
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].set_title('Low-fidelity data', fontsize=16)
axs[0].scatter(lf_data[:, 0], lf_data[:, 1], s=15, c='orange')
axs[1].set_title('High-fidelity data', fontsize=16)
axs[1].scatter(hf_data[:, 0], hf_data[:, 1], s=15, c='red')

for ax in axs:
    ax.set_xlabel(r"$u_1$", fontsize=16)
    ax.grid(True)
    ax.set_xlim(-1.5, 2.)
    ax.set_ylim(-1.5, 2.)
    
axs[0].set_ylabel(r"$u_2$", fontsize=16, rotation=0, labelpad=20)
```

![Example-Data](figures/example-datasets.png)


#### 2.2 Use of `Graph` and `MultiFidelityModel`

Let's now use a multi-fidelity model to transform the graph's nodes. We instantiate a `Graph` with the low-fidelity data as nodes attributes. Thereafter, we pick 10 random indices and consider the corresponding high-fidelity data. This data will be used to update the low-fidelity graph.

```python
## Multi fidelity model

# Only a small set of high-fidelity data (10 points) is used to train the model. 
# The rest of the high-fidelity data is used for visual validation only.
n_lf = lf_data.shape[0]
n_hf = 10

hf_train_inds = np.random.choice(n_lf, n_hf)
hf_data_train = hf_data[hf_train_inds, :]

# Create the graph
graph_config = {
    'metric': 'euclidean',
    'dist_space': 'ambient',
    'method': 'full',
    'corr_scale': None,
    'k_adj': 11,
    'p': 0.5,
    'q': 0.5,
}

graph_lf = Graph(
    data=lf_data,
    **graph_config,
)

# Initialize the model
model_config = {
    'sigma': noise_scale_hf,
}
model = MultiFidelityModel(**model_config)

# Compute multi-fidelity data
mf_data, mf_covar_mat, mf_var = model.transform(
    graph_lf,
    hf_data_train,
    hf_train_inds,
)
model.summary()
```

Output:

```python
=============================================
Model Configuration:
=============================================
sigma              : 0.01
beta               : 2
kappa              : 0.001
omega              : 608.5037816174225
method             : full
spectrum_cutoff    : None
tau                : 0.0012819419490994527
=============================================
```

#### 2.3 Visualize the results

We can now plot the low- and high-fidleity data, together with the new multi-fidelity estimates:

```python
## Plot results
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Low-fidelity data
axs[0].set_title("Low-fidelity data", fontsize=16)
axs[0].scatter(lf_data[:, 0], lf_data[:, 1], color='orange', s=15, label='LF data')
axs[0].scatter(lf_data[hf_train_inds, 0], lf_data[hf_train_inds, 1], s=15, color='blue', label='Selected LF data')
axs[0].scatter(hf_data_train[:, 0], hf_data_train[:, 1], color='red', s=15, label='HF training data')
axs[0].legend()
for i in range(len(hf_train_inds)):
    axs[0].plot(
        [lf_data[hf_train_inds[i], 0], hf_data_train[i, 0]],
        [lf_data[hf_train_inds[i], 1], hf_data_train[i, 1]],
        color='grey',
        linewidth=1.,
        alpha=0.75,
    )

# Multi-fidelity data
axs[1].set_title("Multi-fidelity data", fontsize=16)
axs[1].scatter(mf_data[:, 0], mf_data[:, 1], s=15, color='green')

# High-fidelity data
axs[2].set_title("High-fidelity data", fontsize=16)
axs[2].scatter(hf_data[:, 0], hf_data[:, 1], s=15, color='red')

for ax in axs:
    ax.set_xlabel(r"$u_1$", fontsize=16)
    ax.grid(True)
    ax.set_xlim(-1.5, 2.)
    ax.set_ylim(-1.5, 2.)
axs[0].set_ylabel(r"$u_2$", fontsize=16, rotation=0, labelpad=20)
```

![Example-1](figures/example-1.png)


### 3. Spectral clustering

The `Graph` class has also a built-in functionality to perform spectral clustering of the nodes. The idea is to embed each node in the graph Laplacian eigenfunction space, and then use a standard clustering technique, e.g. K-means. Here's an example of using the `cluster()` method:

```python
## Perform clustering on the graph to find the high-fidelity indices
inds_centroids, labels = graph_lf.cluster(n=n_hf)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_title("Spectral clustering of low-fidleity data", fontsize=16)
ax.scatter(lf_data[:, 0], lf_data[:, 1], s=25, c=labels, cmap='tab10')
ax.scatter(lf_data[inds_centroids, 0], lf_data[inds_centroids, 1], color='red', s=50, label='Centroids')

ax.legend()
ax.set_xlabel(r"$u_1$", fontsize=16)
ax.set_ylabel(r"$u_2$", fontsize=16, rotation=0, labelpad=20)
ax.grid(True)
```

![Example Clustering](figures/example-clustering.png)

#### 3.1 High-fidelity data acquisition policy and `fit_transform()` method

We can leverage the graph clustering to a define a more effective high-fidelity data acquisition policy, rather than just picking random points as above.
Specifically, we determine the centroids of the clusters arising in the low-fidelity data and acquire their high-fidelity counterpart. That is,

```python
## Aquire high-fidelity data at the centroids location
hf_data_train = hf_data[inds_centroids, :]
```

Let's now use the model again with this new selection strategy. Further, we use the `fit_transform()` method of the `MultiFidelityModel` class. This computes the regularization strength parameter so that the multi-fiedlity estimates exhibit a specified level of uncertainty. The default is 3 times the high-fidelity noise/uncertainty.

```python
## Initialize the model
model_config = {
    'sigma': noise_scale_hf,
}
model = MultiFidelityModel(**model_config)

## Fit and transform to get multi-fidelity data
# This will find the value of hyperparameter kappa that leads to given level of uncertainty 
# in the multi-fidelity estimates.
mf_data, mf_covar_mat, mf_var, loss_history, kappa_history = model.fit_transform(
    graph_lf,
    hf_data_train,
    inds_centroids,
    r=3,  # multi-to-high-fidelity uncertainty ratio
    maxiter=200,
    step_size=10,
    step_decay_rate=1.05,
    ftol=1e-6,
    gtol=1e-8,
    verbose=False,
)

# Plot the model final configuration and loss history
model.summary()
plot_loss_and_kappa(loss_history, kappa_history)
```

![Example Clustering](figures/example-kappa-loss-hist.png)

Let's take a look at the results. We notice how the multi-fidelity data resulting from a high-fidelity data acquisition strategy based on clustering are closer to the "underlying truth", i.e. the high-fielity dataset (in red, on the right). Note that the model had only access to the high-fidelity data corresponding to the clusters centroids (shown on the left-most plot, "HF training data"). 

```python
## Plot results
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Low-fidelity data
axs[0].set_title("Low-fidelity data", fontsize=16)
axs[0].scatter(lf_data[:, 0], lf_data[:, 1], color='orange', s=15, label='LF data')
axs[0].scatter(lf_data[inds_centroids, 0], lf_data[inds_centroids, 1], s=15, color='blue', label='Selected LF data')
axs[0].scatter(hf_data_train[:, 0], hf_data_train[:, 1], color='red', s=15, label='HF training data')
axs[0].legend()
for i in range(len(inds_centroids)):
    axs[0].plot(
        [lf_data[inds_centroids[i], 0], hf_data_train[i, 0]],
        [lf_data[inds_centroids[i], 1], hf_data_train[i, 1]],
        color='grey',
        linewidth=1.,
        alpha=0.75,
    )

# Multi-fidelity data
axs[1].set_title("Multi-fidelity data", fontsize=16)
axs[1].scatter(mf_data[:, 0], mf_data[:, 1], s=15, color='green')

# High-fidelity data
axs[2].set_title("High-fidelity data", fontsize=16)
axs[2].scatter(hf_data[:, 0], hf_data[:, 1], s=15, color='red')

for ax in axs:
    ax.set_xlabel(r"$u_1$", fontsize=16)
    ax.grid(True)
    ax.set_xlim(-1.5, 2.)
    ax.set_ylim(-1.5, 2.)
axs[0].set_ylabel(r"$u_2$", fontsize=16, rotation=0, labelpad=20)
```

![Example-2](figures/example-2.png)

### 4. Uncertainty Quantification

The model provides also the variance of each multi-fidelity esimate, which can be interpreted as an uncertainty measure.
We can visualize it for this simple 2-d example by coloring each data point based on the value of variance:

```python
## Plot the variance of multi-fidelity esimates
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
scatter = ax.scatter(mf_data[:, 0], mf_data[:, 1], c=mf_var, s=20,)
ax.scatter(mf_data[inds_centroids, 0], mf_data[inds_centroids, 1], c='red', s=50, label='HF training data')
ax.set_title("Variance of multi-fidelity estimates", fontsize=16)
ax.set_xlabel(r"$u_1$", fontsize=16)
ax.set_ylabel(r"$u_2$", fontsize=16, rotation=0, labelpad=20)
fig.colorbar(scatter, ax=ax,)
ax.legend(loc="upper left", fontsize=12)
ax.grid(True)
```

![Example-variance](figures/example-variance.png)


## Project Structure

```bash
spectral-multifidelity/
│
├── data/                  # Contains datasets for experiments
├── specmf/                # Main code
│   ├── __init__.py        # Initialization file
│   ├── models.py          # Contains graph and model classes
│   ├── preprocess.py      # Data preprocessing utilities
│   ├── graph_core.py      # GraphCore class
│   ├── utils.py           # Utility functions
├── notebooks/             # Jupyter notebooks with examples and experiments
│   └── experiments/       # Experiments with datasets in "data/"
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
