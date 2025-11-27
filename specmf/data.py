"""Data loading utilities for spectral multi-fidelity datasets."""

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


def load_data(
    dataset_name: str,
    preprocess: bool = True,
    normalize: bool = True,
    flatten: bool = True,
    return_normalization_vars: bool = False,
    return_mask: bool = False,
    data_path: str | Path = _data_path,
) -> tuple[np.ndarray, np.ndarray] | list:
    """Load dataset specified by dataset_name.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load.
    preprocess : bool, optional
        Whether to preprocess the data. Default is True.
    normalize : bool, optional
        Whether to normalize the data. Default is True.
    flatten : bool, optional
        Whether to flatten the data. Default is True.
    return_normalization_vars : bool, optional
        Whether to return normalization variables (mean and scale).
        Default is False.
    return_mask : bool, optional
        Whether to return mask computed during preprocessing. Masks out
        NaNs, infs, and duplicates. Default is False.
    data_path : str or Path, optional
        Path to the data directory. Default is the package data dir.

    Returns
    -------
    tuple or list
        Low- and high-fidelity data matrices (X_LF, X_HF), optionally
        with normalization variables and mask.

    Raises
    ------
    ValueError
        If dataset_name is invalid or dimension requirements not met.
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
        valid_names = list(loaders.keys())
        raise ValueError(
            f"Invalid dataset name. Expected one of {valid_names}, "
            f"got {dataset_name} instead."
        )

    X_LF, X_HF = loaders[dataset_name](data_path)
    if X_LF.ndim != X_HF.ndim:
        raise ValueError(
            f"Data matrices must have the same dimensions, "
            f"got {X_LF.ndim} and {X_HF.ndim}."
        )
    if X_LF.ndim != 3 or X_HF.ndim != 3:
        raise ValueError(
            f"Data matrix must be 3-dimensional, "
            f"got shape {X_LF.shape} and {X_HF.shape}."
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
                "Normalization variables are not computed since "
                "normalization is not performed. "
                "Set normalize=True to compute normalization variables."
            )
        results.append(normalization_vars)
    if return_mask:
        if not preprocess:
            raise ValueError(
                "Mask is not computed since preprocessing is not "
                "performed. Set preprocess=True to compute mask."
            )
        results.append(mask)

    return results


def _elasticity_displacement_data(
    data_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load elasticity displacement dataset.

    Parameters
    ----------
    data_path : Path
        Path to the data directory.

    Returns
    -------
    tuple of np.ndarray
        Low- and high-fidelity data matrices.
    """
    logger.info("Loading elasticity displacement data...")
    uy_lf_path = data_path / "elasticity_displacement" / "UY_LF.npy"
    uy_hf_path = data_path / "elasticity_displacement" / "UY_HF.npy"
    X_LF = np.load(uy_lf_path)
    X_HF = np.load(uy_hf_path)
    return X_LF, X_HF


def _darcy_flow_data(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load Darcy flow dataset.

    Parameters
    ----------
    data_path : Path
        Path to the data directory.

    Returns
    -------
    tuple of np.ndarray
        Low- and high-fidelity data matrices.
    """
    logger.info("Loading Darcy flow data...")
    X_LF = np.load(data_path / "darcy" / "X_LF.npy")
    X_HF = np.load(data_path / "darcy" / "X_HF.npy")
    return X_LF, X_HF


def _elasticity_traction_data(
    data_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load elasticity traction dataset.

    Parameters
    ----------
    data_path : Path
        Path to the data directory.

    Returns
    -------
    tuple of np.ndarray
        Low- and high-fidelity data matrices with added dimension.
    """
    logger.info("Loading elasticity traction data...")
    s22_lf_path = data_path / "elasticity_traction" / "S22_LF.npy"
    s22_hf_path = data_path / "elasticity_traction" / "S22_HF.npy"
    X_LF = np.load(s22_lf_path)
    X_HF = np.load(s22_hf_path)
    X_LF = X_LF[:, np.newaxis, :]
    X_HF = X_HF[:, np.newaxis, :]
    return X_LF, X_HF


def _beam_data(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load beam dataset.

    Parameters
    ----------
    data_path : Path
        Path to the data directory.

    Returns
    -------
    tuple of np.ndarray
        Low- and high-fidelity data matrices with added dimension.
    """
    logger.info("Loading beam data...")
    data = np.load(data_path / "beam" / "beam-data.npz")
    X_LF = data["beam_yL"].T
    X_HF = data["beam_yH"].T
    return X_LF[:, np.newaxis, :], X_HF[:, np.newaxis, :]


def _cavity_flow_data(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load cavity flow dataset.

    Parameters
    ----------
    data_path : Path
        Path to the data directory.

    Returns
    -------
    tuple of np.ndarray
        Low- and high-fidelity data matrices with added dimension.
    """
    logger.info("Loading cavity flow data...")
    data = np.load(data_path / "cavity" / "cavity-data.npz")
    X_LF = data["cav_yL"].T
    X_HF = data["cav_yH"].T
    return X_LF[:, np.newaxis, :], X_HF[:, np.newaxis, :]
