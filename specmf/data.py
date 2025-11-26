# Description: This module provides utility functions to load datasets.
import logging
from pathlib import Path

import numpy as np

from specmf.preprocess import (
    flatten_datasets,
    normalize_dataset,
    preprocess_data,
)


# Setup logging
logger = logging.getLogger(__name__)

_data_path = Path(__file__).parent.parent / "data"


# Data loading
def load_data(
    dataset_name: str,
    preprocess: bool = True,
    normalize: bool = True,
    flatten: bool = True,
    return_normalization_vars: bool = False,
    return_mask: bool = False,
    data_path: str | Path = _data_path,
) -> tuple[np.ndarray, np.ndarray] | list:
    """
    Load dataset specified by dataset_name.

    Parameters:
    - dataset_name (str): Name of the dataset to load.
    - preprocess (bool): Whether to preprocess the data. Preprocessing functions live in specmf.preprocess module.
    - normalize (bool): Whether to normalize the data. Normalization functions live in specmf.preprocess module.
    - flatten (bool): Whether to flatten the data. Flattening functions live in specmf.preprocess module.
    - return_normalization_vars (bool): Whether to return normalization variables, e.g. mean and scale.
    - return_mask (bool): Whether to return mask computed during preprocessing;
        if applied to the data, it masks out NaNs, infs, and duplicates.
    - data_path (str): Path to the data directory. Default is "*/spectral-multifidelity/data/".

    Returns:
    - tuple: Low- and high-fidelity data matrix X_LF and X_HF, optionally with normalization vars and mask.
    """
    # Convert to Path if string
    if isinstance(data_path, str):
        data_path = Path(data_path)

    loaders = {
        "elasticity-displacement": _elasticity_displacement_data,
        "darcy-flow": _darcy_flow_data,
        "elasticity-traction": _elasticity_traction_data,
        "beam": _beam_data,
        "cavity-flow": _cavity_flow_data,
    }

    if dataset_name not in loaders:
        raise ValueError(
            f"Invalid dataset name. Expected one of {loaders.keys()}, got {dataset_name} instead."
        )

    X_LF, X_HF = loaders[dataset_name](data_path)
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
def _elasticity_displacement_data(
    data_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    logger.info("Loading elasticity displacement data...")
    X_LF = np.load(data_path / "elasticity_displacement" / "UY_LF.npy")
    X_HF = np.load(data_path / "elasticity_displacement" / "UY_HF.npy")
    # return preprocess_data(X_LF, X_HF)
    return X_LF, X_HF


def _darcy_flow_data(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    logger.info("Loading Darcy flow data...")
    print(data_path)
    X_LF = np.load(data_path / "darcy" / "X_LF.npy")
    X_HF = np.load(data_path / "darcy" / "X_HF.npy")
    # return preprocess_data(X_LF, X_HF)
    return X_LF, X_HF


def _elasticity_traction_data(
    data_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    logger.info("Loading elasticity traction data...")
    X_LF = np.load(data_path / "elasticity_traction" / "S22_LF.npy")
    X_HF = np.load(data_path / "elasticity_traction" / "S22_HF.npy")
    X_LF = X_LF[:, np.newaxis, :]
    X_HF = X_HF[:, np.newaxis, :]
    return X_LF, X_HF


def _beam_data(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    logger.info("Loading beam data...")
    data = np.load(data_path / "beam" / "beam-data.npz")
    X_LF = data["beam_yL"].T
    X_HF = data["beam_yH"].T
    return X_LF[:, np.newaxis, :], X_HF[:, np.newaxis, :]


def _cavity_flow_data(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    logger.info("Loading cavity flow data...")
    data = np.load(data_path / "cavity" / "cavity-data.npz")
    X_LF = data["cav_yL"].T
    X_HF = data["cav_yH"].T
    return X_LF[:, np.newaxis, :], X_HF[:, np.newaxis, :]
