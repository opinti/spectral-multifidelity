"""Utility functions for spectral multi-fidelity modeling."""

import logging
from pathlib import Path

import numpy as np
import yaml
from sklearn.cluster import KMeans


# Setup logging
logger = logging.getLogger(__name__)


def ordered_eig(
    X: np.ndarray, symmetric: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Perform eigenvalue decomposition with sorted eigenvalues.

    Parameters
    ----------
    X : np.ndarray
        Input matrix for eigenvalue decomposition.
    symmetric : bool, optional
        If True, assumes X is symmetric/Hermitian and uses eigh
        for better performance and numerical stability. Default True.

    Returns
    -------
    tuple of np.ndarray
        Sorted eigenvalues and corresponding eigenvectors.
    """
    if symmetric:
        # Use eigh for symmetric matrices (faster and more stable)
        eigvals, eigvecs = np.linalg.eigh(X)
    else:
        eigvals, eigvecs = np.linalg.eig(X)

    sorting_ind = np.argsort(np.abs(eigvals))
    return eigvals[sorting_ind], eigvecs[:, sorting_ind]


# Constants
DEFAULT_KMEANS_N_INIT = 15
DEFAULT_RANDOM_STATE = 42


def spectral_clustering(
    eigvecs: np.ndarray,
    n: int,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform spectral clustering and return cluster info.

    Parameters
    ----------
    eigvecs : np.ndarray
        Eigenvectors of the graph Laplacian matrix.
    n : int
        Number of clusters.
    random_state : int, optional
        Random state for reproducibility. Default is 42.

    Returns
    -------
    tuple of np.ndarray
        Indices of cluster centroids and labels of data points.
    """
    P = np.real(eigvecs[:, :n])
    kmeans = KMeans(
        n_clusters=n,
        random_state=random_state,
        n_init=DEFAULT_KMEANS_N_INIT,
    ).fit(P)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    inds_centroids = np.argmin(
        np.linalg.norm(P[:, np.newaxis] - cluster_centers, axis=2),
        axis=0,
    )

    return inds_centroids, labels


def rearrange(X: np.ndarray, first_inds: list[int]) -> np.ndarray:
    """Rearrange rows of matrix with specified indices on top.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix.
    first_inds : list of int
        List of indices to be placed at the top.

    Returns
    -------
    np.ndarray
        Rearranged data matrix.
    """
    first_inds_set = set(first_inds)
    if not first_inds_set.issubset(set(range(X.shape[0]))):
        raise ValueError("Invalid indices in first_inds.")

    other_inds = [i for i in range(X.shape[0]) if i not in first_inds_set]
    return np.vstack([X[first_inds, :], X[other_inds, :]])


def error_analysis(
    x_lf: np.ndarray,
    x_mf: np.ndarray,
    x_hf: np.ndarray,
    component_wise: bool = False,
    return_values: bool = False,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute relative error between LF, MF, and HF data.

    Parameters
    ----------
    x_lf : np.ndarray
        Low-fidelity data of shape (n_samples, n_features).
    x_mf : np.ndarray
        Multi-fidelity data of shape (n_samples, n_features).
    x_hf : np.ndarray
        High-fidelity reference data (n_samples, n_features).
    component_wise : bool, optional
        If True, compute component-wise error along axis=0.
        Default is False.
    return_values : bool, optional
        If True, return the computed errors. Default is False.
    verbose : bool, optional
        If True, print the computed errors. Default is True.

    Returns
    -------
    None or tuple of np.ndarray
        If return_values is True, returns tuple (error_lf, error_mf).
        Otherwise, returns None.
    """

    if component_wise:
        hf_mean_axis0 = np.mean(np.abs(x_hf), axis=0)
        error_lf_i = 100 * np.abs(x_lf - x_hf) / hf_mean_axis0
        error_lf_mean = np.mean(error_lf_i, axis=0)
        error_lf_std = np.std(error_lf_i, axis=0)

        error_mf_i = 100 * np.abs(x_mf - x_hf) / hf_mean_axis0
        error_mf_mean = np.mean(error_mf_i, axis=0)
        error_mf_std = np.std(error_mf_i, axis=0)

        error_label = "Component-wise mean"
    else:
        error_lf_i = (
            100
            * np.linalg.norm(x_lf - x_hf, axis=1)
            / np.mean(np.linalg.norm(x_hf, axis=1))
        )
        error_lf_mean = np.array([np.mean(error_lf_i)])
        error_lf_std = np.array([np.std(error_lf_i)])

        error_mf_i = (
            100
            * np.linalg.norm(x_mf - x_hf, axis=1)
            / np.mean(np.linalg.norm(x_hf, axis=1))
        )
        error_mf_mean = np.array([np.mean(error_mf_i)])
        error_mf_std = np.array([np.std(error_mf_i)])

        error_label = "Mean"

    if verbose:
        max_width = max(
            len(f"{mean} ({std})")
            for mean, std in zip(
                np.round(error_lf_mean, 2),
                np.round(error_lf_std, 2),
                strict=False,
            )
        )

        lf_formatted = " ".join([
            f"{mean} ({std})".ljust(max_width)
            for mean, std in zip(
                np.round(error_lf_mean, 2),
                np.round(error_lf_std, 2),
                strict=False,
            )
        ])
        mf_formatted = " ".join([
            f"{mean} ({std})".ljust(max_width)
            for mean, std in zip(
                np.round(error_mf_mean, 2),
                np.round(error_mf_std, 2),
                strict=False,
            )
        ])
        drop_formatted = " ".join([
            f"{drop}%".ljust(max_width)
            for drop in np.round(
                100 * (error_lf_mean - error_mf_mean) / error_lf_mean, 2
            )
        ])

        suptitle = (
            f"{error_label} relative L2 errors and percentage error drop"
        )
        print(suptitle)
        print("-" * len(suptitle))
        print(f"Error LF:   {lf_formatted}")
        print(f"Error MF:   {mf_formatted}")
        print(f"[%] drop:   {drop_formatted}")

    if return_values:
        return error_lf_mean, error_mf_mean


def load_model_config(
    config_path: str,
    dataset_name: str,
    return_n_HF: bool = False,
) -> dict | tuple[dict, int | None]:
    """Load model configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    dataset_name : str
        Name of the dataset to fetch configuration for.
    return_n_HF : bool, optional
        Flag indicating if n_HF should be returned. Default is False.

    Returns
    -------
    dict or tuple
        Model configuration, optionally with n_HF if return_n_HF=True.
    """
    try:
        with Path.open(config_path) as file:
            model_config_data = yaml.safe_load(file)

        if dataset_name not in model_config_data:
            raise KeyError(
                f"Dataset '{dataset_name}' not found in the "
                f"configuration file."
            )

        dataset_cfg = model_config_data[dataset_name]
        model_config = dataset_cfg.get("model_config")

        if return_n_HF:
            n_HF = dataset_cfg.get("n_HF", None)
            return model_config, n_HF

        return model_config

    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}.")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise
