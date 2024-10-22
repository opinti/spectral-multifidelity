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
For end-to-end examples, please check the experiment notebooks in `notebooks/` and `notebooks/experiments/`.

### Example Usage

The main module of the package is `models.py`. Below are examples demonstrating how to use this module.

### 1. Using the `Graph` class

The `Graph` class inherhits from `GraphCore` class, which computes graph-based representations, such as the adjacency matrix and the normalized graph Laplacian. 
Here is an example of how to initialize an instance of the class and compute graph's representation matrices based on the nodes attributes.

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


### 2. Using the `MultiFidelityModel` class

The `MultiFidelityModel` class is designed to perform multi-fidelity modeling. 
It allows to transform all nodes of a graph based on a few more accurate nodes data, i.e. the "high-fidelity" data.


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


#### 2.2 Use of `MultiFidelityModel`

Let's now use a multi-fidelity model to transform the low-fidelity data based on a few high-fidelity data points.
We pick 10 random indices and consider the corresponding high-fidelity data. This data is denoted as 'training data', and will be used to update the full low-fidelity graph.
To do that, we instantiate a `Graph` with the low-fidelity data as nodes attributes.
Thereafter, we use the `tranform()` method of the `MultiFidelityModel()` class to update the graph nodes.
This method uses the low- and high-fidelity data pairs to derive a transformation for all nodes.
We use the default configuration for the model instance, except for the training data noise level, which is set based on the known or assumed noise of the high-fidelity data (`noise_scale_hf`). 
After the transformation is computed and applied, we print a summary of the model configuration and parameters.

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
    'sigma': noise_scale_hf,  # The noise level of the high-fidelity data
}
model = MultiFidelityModel(**model_config)

# Compute multi-fidelity data
mf_data, mf_covar_mat, mf_std = model.transform(
    graph_lf,
    hf_data_train,
    hf_train_inds,
)
model.summary()
```

Output:

```text
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

We can now plot the low- and high-fidelity data, together with the new multi-fidelity estimates:

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


### 3. High-fidelity data acquisition policy

The method uses a small number of low- and high-fielity data pairs to update all low-fidelity data.
In the example above we picked random pairs, but we can design more effective strategies. 
For example, we can cluster the low-fidelity data and decide to 'acquire' high-fidelity data corresponding to the centroids of the clusters.
This will make sure that the acquired high-fidelity data cover the whole graph more uniformly, and that if the graph has specific structures,
these will be preserved better and carried over the multi-fidelity estimates.

#### 3.1 Spectral clustering

The `Graph` class has a built-in `cluster()` method to perform spectral clustering of the nodes.
The idea is embedding the nodes in the graph Laplacian eigenfunction space, and then use a standard clustering technique, e.g. K-means.
This method returns the indices of the clusters centroids and the cluster label of all nodes.

Here's an example of usage of the `cluster()` method:

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

#### 3.1 Use of the `fit_transform()` method

We leverage the graph clustering above to a decide which high-fidelity data to acquire.
Specifically, we select the high-fidelity data that corresponds to the centroids of the low-fidelity graph's clusters. That is,

```python
## Aquire high-fidelity data at the centroids location
hf_data_train = hf_data[inds_centroids, :]
```

Let's now use the model again with this new selection strategy.
Further, to compute the transformation we use the `fit_transform()` method of the `MultiFidelityModel` class. 
This finds the regularization strength parameter `kappa` that results in a specified mean level of uncertainty of the multi-fidelity estimates (`mf_std`).
This is done by specifing the multi-to-high-fidelity uncertainty ratio `r`, whose default velue is 3.

```python
## Initialize the model
model_config = {
    'sigma': noise_scale_hf,
}
model = MultiFidelityModel(**model_config)

## Fit and transform to get multi-fidelity data
# This will find the value of hyperparameter kappa that leads to given level of uncertainty 
# in the multi-fidelity estimates.
mf_data, mf_covar_mat, mf_std, loss_history, kappa_history = model.fit_transform(
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

#### 3.2 Visualize the results

Let's take a look at the results obtained with a high-fidelity data acquisition strategy based on clustering.
We notice how in this case the multi-fidelity data (in blue, in the center figure) are closer to the "underlying truth", i.e. the high-fidelity dataset (in red, on the right).
Note that the model had only access to the high-fidelity data corresponding to the clusters centroids (shown in the left-most plot, "HF training data"). 

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

The model provides also the standard deviation of each multi-fidelity esimate, which can be interpreted as an uncertainty measure.
We can visualize it for this simple 2-d example by coloring each data point based on the value of variance.
As expected, the uncertainty is larger for points that are further from the training data (the red dots), and approaches
the high-fidelity noise level (0.01) closer to them. 

```python
## Plot the standard deviation of multi-fidelity esimates
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
scatter = ax.scatter(mf_data[:, 0], mf_data[:, 1], c=mf_std, s=20,)
ax.scatter(mf_data[inds_centroids, 0], mf_data[inds_centroids, 1], c='red', s=50, label='HF training data')
ax.set_title("Standard deviation of multi-fidelity estimates", fontsize=16)
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
