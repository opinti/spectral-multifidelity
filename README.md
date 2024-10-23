# Graph Laplacian-based Bayesian Multi-fidelity Modeling

## Overview

A python package for graph laplacian-based multi-fidelity modeling.

![Graphical Abstract](figures/graphical-abstract.png)

## Features

- Perform multi-fidelity modeling using Bayesian specmf method.
- Custom `Graph` class for computing graph representations and performing spectral clustering.
- Modeling computations happen in a dedicated `MultiFidelityModel` class.
- Utilities for preprocessing and visualize data in the context of multi-fidelity modeling.
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

- `numpy==1.26.0`
- `scipy==3.8.0`
- `scikit-learn==1.5.1`
- `scikit-optimize==0.8.1`
- `umap-learn==0.5.6`
- `matplotlib==4.2.5`

The package was developed and tested with Python 3.12.

## Usage

The following sections describe different functionalities provided by the package.
For end-to-end examples, please check the experiment notebooks in `notebooks/` and `notebooks/experiments/`.

### Example Usage

The main module of the package is `models.py`. Below are examples demonstrating how to use this module.

### 1. Using the `Graph` class

The `Graph` class inherits from `GraphCore` class, which computes graph-based representations, such as the adjacency matrix and the normalized graph Laplacian.
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
It transforms all nodes of a graph using a subset of more accurate high-fidelity data.

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

We now use a multi-fidelity model to transform the low-fidelity data based on a small subset of high-fidelity points. We randomly select 10 indices and use the corresponding high-fidelity data as 'training data' to update the entire low-fidelity graph. 
To achieve this, we first initialize a `Graph` with the low-fidelity data as node attributes.
Next, we apply the `tranform()` method of the `MultiFidelityModel` class, which uses the low- and high-fidelity data pairs to determine a transformation for all nodes. We maintain the default model configuration, except for the noise level of the training data, which is set according to the known or assumed noise of the high-fidelity data (`noise_scale_hf`). Once the transformation is complete, a summary of the model configuration and parameters is printed.

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

The method updates all low-fidelity data using a small set of low- and high-fidelity data pairs.
In the previous example we selected random pairs, but we can design more effective strategies.
For instance, we can cluster the low-fidelity data and decide to 'acquire' high-fidelity data corresponding to the centroids of the clusters.
This ensures that the high-fidelity data covers the entire graph uniformly and better preserves any underlying structures, leading to more accurate multi-fidelity estimates.

#### 3.1 Spectral clustering

The `Graph` class includes a `cluster()` method for performing spectral clustering on the nodes. 
This involves embedding the nodes in the graph Laplacian eigenfunction space, and then applying a standard clustering technique, e.g. K-means. The method returns both the indices of the cluster centroids and the cluster labels for all nodes.

Here's an example of usage of the `cluster()` method:

```python
## Perform clustering on the graph to find the high-fidelity indices
inds_centroids, labels = graph_lf.cluster(n=n_hf)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_title("Spectral clustering of low-fidelity data", fontsize=16)
ax.scatter(lf_data[:, 0], lf_data[:, 1], s=25, c=labels, cmap='tab10')
ax.scatter(lf_data[inds_centroids, 0], lf_data[inds_centroids, 1], color='red', s=50, label='Centroids')

ax.legend()
ax.set_xlabel(r"$u_1$", fontsize=16)
ax.set_ylabel(r"$u_2$", fontsize=16, rotation=0, labelpad=20)
ax.grid(True)
```

![Example Clustering](figures/example-clustering.png)

#### 3.2 Use of the `fit_transform()` method

We leverage the graph clustering to determine which high-fidelity data to acquire. Specifically, we select the high-fidelity data corresponding to the centroids of the low-fidelity graph's clusters. That is:

```python
## Aquire high-fidelity data at the centroids location
hf_data_train = hf_data[inds_centroids, :]
```

Let’s now apply the model again using this new selection strategy. 
To compute the transformation, we use the `fit_transform()` method of the `MultiFidelityModel` class.
This method finds the regularization parameter `kappa` that achieves a specified mean level of uncertainty in the multi-fidelity estimates (`mf_std`). The uncertainty is controlled by setting the multi-to-high-fidelity uncertainty ratio `r`, with a default value of 3.

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

Let’s examine the results obtained with a high-fidelity data acquisition strategy based on clustering.
In this case, the multi-fidelity data (shown in green, in the center figure) are much closer to the "underlying truth", i.e. the high-fidelity dataset (in red, on the right).
It’s important to note that the model only had access to the high-fidelity data corresponding to the clusters centroids (shown in the left-most plot as "HF training data").

![Example-2](figures/example-transition.gif)

### 4. Uncertainty Quantification

The model also provides the standard deviation for each multi-fidelity estimate, which serves as a measure of uncertainty.
In this simple 2D example, we can visualize the uncertainty by coloring each data point based on its standard deviation.
As expected, uncertainty increases for points further from the training data (the red dots) and approaches the high-fidelity noise level (0.01) for points closer to them.

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
├── setup.py               # Setup file
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation (this file)
```

## Contact

For any questions or issues, please reach out at orazio.pinti@gmail.com.

## Reference

Source code of the paper: 

**Graph Laplacian-based Bayesian Multi-fidelity Modelinge**  
O. Pinti, J. M. Budd, F. Hoffmann, A. A. Oberai.
[arXiv:2409.08211](https://arxiv.org/abs/2409.08211)
