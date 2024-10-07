# Description: This module provides utility functions to load datasets.
import numpy as np
import os
from typing import Tuple
from specmf.preprocess import preprocess_data, normalize_dataset, flatten_datasets

data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/")


# Data loading
def load_data(
    dataset_name: str,
    preprocess: bool = True,
    normalize: bool = True,
    flatten: bool = True,
    return_normalization_vars: bool = False,
    return_mask: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset specified by dataset_name.

    Parameters:
    - dataset_name (str): Name of the dataset to load.

    Returns:
    - tuple: Low- and high-fidelity data matrix X_LF and X_HF.
    """
    loaders = {
        "inclusion-field": _inclusion_field_data,
        "darcy-flow": _darcy_flow_data,
        "inclusion-qoi": _inclusion_qoi_data,
        "beam": _beam_data,
        "cavity": _cavity_data,
    }

    if dataset_name not in loaders:
        raise ValueError(
            f"Invalid dataset name. Expected one of {loaders.keys()}, got {dataset_name} instead."
        )

    X_LF, X_HF = loaders[dataset_name]()
    if X_LF.ndim != X_HF.ndim:
        raise ValueError(
            f"Data matrices must have the same dimensions, got {X_LF.ndim} and {X_HF.ndim}."
        )
    if X_LF.ndim != 3 or X_HF.ndim != 3:
        raise ValueError(
            f"Data matrix must be 3-dimensional, got shape {X_LF.shape} and {X_HF.shape}."
        )

    if preprocess:
        X_LF, X_HF, mask = preprocess_data(X_LF, X_HF)
    if normalize:
        X_LF, X_HF, normalization_vars = normalize_dataset(
            X_LF, X_HF, dataset_name, return_normalization_vars=True
        )
    if flatten:
        X_LF, X_HF = flatten_datasets(X_LF, X_HF)

    results = [X_LF, X_HF]

    if return_normalization_vars:
        if not normalize:
            raise ValueError(
                "Normalization variables are not computed since normalization is not performed. "
                "Set normalize=True to compute normalization variables."
            )
        results.append(normalization_vars)
    if return_mask:
        if not preprocess:
            raise ValueError(
                "Mask is not computed since preprocessing is not performed. "
                "Set preprocess=True to compute mask."
            )
        results.append(mask)

    return results


# Data loaders for different datasets
def _inclusion_field_data() -> Tuple[np.ndarray, np.ndarray]:
    print("Loading inclusion fields data ...")
    X_LF = np.load(os.path.join(data_path, "solid_inclusion/UY_LF.npy"))
    X_HF = np.load(os.path.join(data_path, "solid_inclusion/UY_HF.npy"))
    # return preprocess_data(X_LF, X_HF)
    return X_LF, X_HF


def _darcy_flow_data() -> Tuple[np.ndarray, np.ndarray]:
    print("Loading Darcy flow data ...")
    X_LF = np.load(os.path.join(data_path, "darcy/X_LF.npy"))
    X_HF = np.load(os.path.join(data_path, "darcy/X_HF.npy"))
    # return preprocess_data(X_LF, X_HF)
    return X_LF, X_HF


def _inclusion_qoi_data() -> Tuple[np.ndarray, np.ndarray]:
    print("Loading inclusion QoIs data ...")
    X_LF = np.load(os.path.join(data_path, "solid_inclusion_qoi/S22_LF.npy"))
    X_HF = np.load(os.path.join(data_path, "solid_inclusion_qoi/S22_HF.npy"))
    X_LF = X_LF[:, np.newaxis, :]
    X_HF = X_HF[:, np.newaxis, :]
    return X_LF, X_HF


def _beam_data() -> Tuple[np.ndarray, np.ndarray]:
    print("Loading beam data ...")
    data = np.load(os.path.join(data_path, "beam/beam-data.npz"))
    X_LF = data["beam_yL"].T
    X_HF = data["beam_yH"].T
    return X_LF[:, np.newaxis, :], X_HF[:, np.newaxis, :]


def _cavity_data() -> Tuple[np.ndarray, np.ndarray]:
    print("Loading cavity data ...")
    data = np.load(os.path.join(data_path, "cavity/cavity-data.npz"))
    X_LF = data["cav_yL"].T
    X_HF = data["cav_yH"].T
    return X_LF[:, np.newaxis, :], X_HF[:, np.newaxis, :]
