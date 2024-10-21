import numpy as np
from typing import Tuple, Dict


def preprocess_data(
    X_LF: np.ndarray, X_HF: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess datasets: removes NaN, Inf values, and duplicates.

    Parameters:
    - X_LF (np.ndarray): Low-fidelity data matrix (n_dim1, n_dim2, n_samples).
    - X_HF (np.ndarray): High-fidelity data matrix (n_dim1, n_dim2, n_samples).

    Returns:
    - Tuple: Preprocessed low- and high-fidelity matrices and a global mask. The
        global mask is a boolean array that indicates the valid points in the datasets.
    """
    X_LF, X_HF, mask_clean = _clean_data(X_LF, X_HF)
    X_LF, X_HF, mask_uniques = _remove_duplicates(X_LF, X_HF)

    mask_global = np.zeros_like(mask_clean, dtype=bool)
    mask_global[np.where(mask_clean)[0][mask_uniques]] = True

    return X_LF, X_HF, mask_global


def _clean_data(
    X1: np.ndarray, X2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove snapshots from datasets if NaN or Inf values are present.

    Parameters:
    - X1 (np.ndarray): Low-fidelity data matrix.
    - X2 (np.ndarray): High-fidelity data matrix.

    Returns:
    - Tuple: Cleaned data matrices and a mask of valid points.
    """
    if X1.ndim != X2.ndim or X1.ndim != 3:
        raise ValueError(
            f"Input matrices must be 3D with matching shapes, got {X1.shape} and {X2.shape}."
        )

    X1_flat, X2_flat = _flatten_snapshots(X1), _flatten_snapshots(X2)
    mask = np.any(
        np.isnan(X1_flat) | np.isinf(X1_flat) | np.isnan(X2_flat) | np.isinf(X2_flat),
        axis=1,
    )
    mask_clean = ~mask

    return (
        _unflatten_snapshots(X1_flat[mask_clean], X1.shape[:2]),
        _unflatten_snapshots(X2_flat[mask_clean], X2.shape[:2]),
        mask_clean,
    )


def _remove_duplicates(
    X_LF: np.ndarray, X_HF: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove duplicate snapshots from the datasets.

    Parameters:
    - X_LF (np.ndarray): Low-fidelity data matrix.
    - X_HF (np.ndarray): High-fidelity data matrix.

    Returns:
    - Tuple: Unique low- and high-fidelity data matrices and mask of unique points.
    """
    X_LF_flat, X_HF_flat = _flatten_snapshots(X_LF), _flatten_snapshots(X_HF)

    _, unique_idxs = np.unique(X_LF_flat, axis=0, return_index=True)
    unique_idxs = np.sort(unique_idxs)

    mask_uniques = np.zeros(X_LF.shape[-1], dtype=bool)
    mask_uniques[unique_idxs] = True

    return (
        _unflatten_snapshots(X_LF_flat[unique_idxs], X_LF.shape[:2]),
        _unflatten_snapshots(X_HF_flat[unique_idxs], X_HF.shape[:2]),
        mask_uniques,
    )


def _remove_outliers(
    X: np.ndarray, Y: np.ndarray, max_z_score: float = 3.0, axis: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers based on z-score.

    Parameters:
    - X (np.ndarray): Reference data matrix.
    - Y (np.ndarray): Dependant data matrix.
    - max_z_score (float): Maximum allowed z-score.
    - axis (int): Axis along which to calculate the z-score.

    Returns:
    - Tuple: Cleaned reference and dependent matrices.
    """
    mean = np.mean(X, axis=axis)
    std_dev = np.std(X, axis=axis)

    if axis == 0:
        z_scores = np.abs((X - mean) / std_dev)
    else:
        z_scores = np.abs((X.T - mean) / std_dev).T

    other_axis = 1 if axis == 0 else 0
    outliers = np.any(z_scores > max_z_score, axis=other_axis)

    # Remove outliers
    if axis == 0:
        X_clean = X[~outliers, :]
        Y_clean = Y[~outliers, :]
    else:
        X_clean = X[:, ~outliers]
        Y_clean = Y[:, ~outliers]

    return X_clean, Y_clean


# Data normalizers
def normalize_dataset(
    X_LF: np.ndarray,
    X_HF: np.ndarray,
    dataset_name: str,
    return_normalization_vars: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Normalize datasets based on the given dataset name.

    Parameters:
    - X_LF (np.ndarray): Low-fidelity data matrix.
    - X_HF (np.ndarray): High-fidelity data matrix.
    - dataset_name (str): Name of the dataset.

    Returns:
    - Tuple: Normalized datasets and (optionally) normalization variables.
    """
    normalizers = {
        "elasticity-displacement": _normalize_elasticity_displacement,
        "darcy-flow": _normalize_darcy_flow,
        "elasticity-traction": _normalize_elasticity_traction,
    }

    if dataset_name not in normalizers:
        raise ValueError(
            f"Invalid dataset name. Available options: {list(normalizers.keys())}."
        )

    return normalizers[dataset_name](X_LF, X_HF, return_normalization_vars)


def _normalize_elasticity_displacement(
    X_LF: np.ndarray, X_HF: np.ndarray, return_normalization_vars: bool
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Normalize 'elasticity-displacement' dataset.
    """
    n_dim = X_LF.shape[1]
    UY_mean = np.linspace(0, np.mean(X_LF[-1], axis=0), n_dim)[:, np.newaxis, :]
    X_LF -= UY_mean
    X_HF -= UY_mean

    X_LF_scale = np.max(np.abs(X_LF), axis=(0, 1)) + 1e-8
    X_LF /= X_LF_scale
    X_HF /= X_LF_scale

    if return_normalization_vars:
        return X_LF, X_HF, {"X_mean": UY_mean, "X_scale": X_LF_scale}
    return X_LF, X_HF


def _normalize_darcy_flow(
    X_LF: np.ndarray, X_HF: np.ndarray, return_normalization_vars: bool
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Normalize 'darcy-flow' dataset.
    """
    X_LF_mean, X_HF_mean = np.mean(X_LF, axis=(0, 1)), np.mean(X_HF, axis=(0, 1))
    X_LF -= X_LF_mean
    X_HF -= X_HF_mean

    X_LF_scale = np.max(np.abs(X_LF), axis=(0, 1)) + 1e-8
    X_LF /= X_LF_scale
    X_HF /= X_LF_scale

    if return_normalization_vars:
        return (
            X_LF,
            X_HF,
            {"X_LF_mean": X_LF_mean, "X_HF_mean": X_HF_mean, "X_scale": X_LF_scale},
        )
    return X_LF, X_HF


def _normalize_elasticity_traction(
    X_LF: np.ndarray, X_HF: np.ndarray, return_normalization_vars: bool
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Normalize 'elasticity-traction' dataset, reshaped to 3D for compatibility.

    Parameters:
    - X_LF (np.ndarray): Low-fidelity data matrix.
    - X_HF (np.ndarray): High-fidelity data matrix.

    Returns:
    - Tuple: Normalized inclusion QoI matrices and (optionally) normalization variables.
    """

    def _generate_inclusion_qoi(X: np.ndarray) -> np.ndarray:
        out_dim = 4
        step = X.shape[0] // out_dim
        U = np.array(
            [
                np.trapz(
                    X[i * step : (i + 1) * step, :], dx=0.001, axis=0  # noqa: E203
                )
                for i in range(out_dim)
            ]
        )
        U = np.vstack((U, np.max(X, axis=0)))
        return U

    X_LF, X_HF = X_LF.squeeze(axis=1), X_HF.squeeze(axis=1)

    X_LF, X_HF = _generate_inclusion_qoi(X_LF), _generate_inclusion_qoi(X_HF)
    X_LF, X_HF = _remove_outliers(X_LF, X_HF, max_z_score=3.5, axis=1)

    X_LF_mean, X_LF_std = np.mean(X_LF, axis=1), np.std(X_LF, axis=1)
    X_LF = _normalize(X_LF.T, X_LF_mean, X_LF_std).T
    X_HF = _normalize(X_HF.T, X_LF_mean, X_LF_std).T

    if return_normalization_vars:
        return (
            X_LF[:, np.newaxis, :],
            X_HF[:, np.newaxis, :],
            {"X_mean": X_LF_mean, "X_scale": X_LF_std},
        )

    return X_LF[:, np.newaxis, :], X_HF[:, np.newaxis, :]


def _normalize(X: np.ndarray, X_mean: np.ndarray, X_scale: np.ndarray) -> np.ndarray:
    """
    Scale the input matrix of fields based on the mean and scale values.
    """
    return (X - X_mean) / (X_scale + 1e-8)


# Flatten and unflatten data
def flatten_datasets(
    X_LF: np.ndarray, X_HF: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten the input data matrices into 2D arrays.

    Parameters:
    - X_LF (np.ndarray): Low-fidelity data matrix.
    - X_HF (np.ndarray): High-fidelity data matrix.

    Returns:
    - Tuple: Flattened low- and high-fidelity data matrices.
    """
    return _flatten_snapshots(X_LF), _flatten_snapshots(X_HF)


def _flatten_snapshots(X: np.ndarray) -> np.ndarray:
    """
    Flatten the snapshots in the data matrix.

    Parameters:
    - X (np.ndarray): Data matrix of shape (n_dim1, n_dim2, n_samples).

    Returns:
    - Flattened data matrix of shape (n_samples, n_dim1 * n_dim2).
    """
    if X.ndim != 3:
        raise ValueError(f"Input matrix must be 3D. Got shape {X.shape}.")

    n_samples = X.shape[-1]
    return np.array([X[:, :, i].flatten() for i in range(n_samples)])


def _unflatten_snapshots(X: np.ndarray, shape_X: Tuple[int, int]) -> np.ndarray:
    """
    Unflatten 2D data back into 3D snapshots.

    Parameters:
    - X (np.ndarray): Flattened data matrix.
    - shape_X (Tuple[int, int]): Original dimensions of snapshots. If not provided, the
        function will assume square snapshots.


    Returns:
    - 3D unflattened data matrix.
    """
    if X.ndim != 2:
        raise ValueError(f"Input matrix must be 2-dimensional, got shape {X.shape}.")

    n_samples = X.shape[0]
    if shape_X is None:
        n_dim1 = int(np.sqrt(X.shape[1]))
        n_dim2 = n_dim1
    else:
        n_dim1, n_dim2 = shape_X

    X_ = np.zeros((n_dim1, n_dim2, n_samples))
    for i in range(n_samples):
        X_[:, :, i] = X[i, :].reshape(n_dim1, n_dim2)
    return X_
