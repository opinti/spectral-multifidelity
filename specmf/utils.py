import numpy as np
from typing import List, Tuple, Union
from sklearn.cluster import KMeans
import yaml
import logging


def ordered_eig(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform eigenvalue decomposition of a matrix X and return the sorted eigenvalues and corresponding eigenvectors.

    Parameters:
    - X (numpy.ndarray): Input matrix for eigenvalue decomposition.

    Returns:
    - Tuple[numpy.ndarray, numpy.ndarray]: Sorted eigenvalues and corresponding eigenvectors.
    """
    eigvals, eigvecs = np.linalg.eig(X)
    sorting_ind = np.argsort(np.abs(eigvals))
    return eigvals[sorting_ind], eigvecs[:, sorting_ind]


def spectral_clustering(
    eigvecs: np.ndarray, n: int, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform spectral clustering and return the indices of the centroids of the clusters.

    Parameters:
    - eigvecs (numpy.ndarray): Eigenvectors of the graph Laplacian matrix.
    - n (int): Number of clusters.

    Returns:
    - tuple: Indices of the centroids of the clusters, labels of the data points.
    """
    P = np.real(eigvecs[:, :n])
    kmeans = KMeans(n_clusters=n, random_state=random_state, n_init=15).fit(P)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    inds_centroids = np.argmin(
        np.linalg.norm(P[:, np.newaxis] - cluster_centers, axis=2), axis=0
    )

    return inds_centroids, labels


def rearrange(X: np.ndarray, first_inds: List[int]) -> np.ndarray:
    """
    Rearrange the rows of the input matrix X so that the indices specified by first_inds are on top.

    Parameters:
    - X (numpy.ndarray): Input data matrix.
    - first_inds (List[int]): List of indices to be placed at the top.

    Returns:
    - numpy.ndarray: Rearranged data matrix.
    """
    first_inds_set = set(first_inds)
    if not first_inds_set.issubset(set(range(X.shape[0]))):
        raise ValueError("Invalid indices in first_inds.")

    other_inds = [i for i in range(X.shape[0]) if i not in first_inds_set]
    return np.vstack([X[first_inds, :], X[other_inds, :]])


def error_analysis(
    x_lf,
    x_mf,
    x_hf,
    component_wise: bool = False,
    return_values: bool = False,
    verbose: bool = False,
) -> Union[None, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute the relative error of low- and multi-fidelity data, with respect
    to referece and high-fidelity data.

    Parameters:
    - x_lf (numpy.ndarray): Low-fidelity data of shape (n_points, n_dim).
    - x_mf (numpy.ndarray): Multi-fidelity data of shape (n_points, n_dim).
    - x_hf (numpy.ndarray): High-fidelity data of shape (n_points, n_dim).
    - component_wise (bool): If True, compute component-wise error.
    - return_values (bool): If True, return the computed errors.
    - verbose (bool): If True, print the computed errors.
    """
    if component_wise:
        if verbose:
            print("Component-wise mean relative l2 errors and Improvement Factor (IF)")
            print("-------------------------------------------------------------------")
        e_lf = (
            100 * np.mean(np.abs(x_lf - x_hf), axis=0) / np.mean(np.abs(x_hf), axis=0)
        )
        e_mf = (
            100 * np.mean(np.abs(x_mf - x_hf), axis=0) / np.mean(np.abs(x_hf), axis=0)
        )
    else:
        if verbose:
            print("Mean relative l2 errors and Improvement Factor (IF)")
            print("----------------------------------------------------")
        e_lf = (
            100
            * np.mean(np.linalg.norm(x_lf - x_hf, axis=1))
            / np.mean(np.linalg.norm(x_hf, axis=1))
        )
        e_mf = (
            100
            * np.mean(np.linalg.norm(x_mf - x_hf, axis=1))
            / np.mean(np.linalg.norm(x_hf, axis=1))
        )

    if verbose:
        print(f"Error LF:         {np.round(e_lf, 2)}")
        print(f"Error MF:         {np.round(e_mf, 2)}")
        print(f"Percentage drop:  {np.round(100 * (e_lf - e_mf) / e_lf, 2)}%")

    if return_values:
        return e_lf, e_mf


def load_model_config(config_path: str, dataset_name: str, return_n_HF: bool = False):
    """
    Load model configuration from a YAML file.

    Parameters:
    - config_path (str): Path to the YAML configuration file.
    - dataset_name (str): Name of the dataset to fetch the configuration for.
    - return_n_HF (bool): Flag indicating if n_HF should be returned. Default is False.

    Returns:
        dict or tuple: Model configuration, optionally with n_HF if return_n_HF is True.
    """
    try:
        with open(config_path, "r") as file:
            model_config_data = yaml.safe_load(file)

        if dataset_name not in model_config_data:
            raise KeyError(
                f"Dataset '{dataset_name}' not found in the configuration file."
            )

        model_config = model_config_data[dataset_name].get("model_config")

        if return_n_HF:
            n_HF = model_config_data[dataset_name].get("n_HF", None)
            return model_config, n_HF

        return model_config

    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}.")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise
