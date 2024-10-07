import numpy as np
from typing import Tuple, Dict


def preprocess_data(
    X_LF: np.ndarray, X_HF: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess datasets: removes NaN, Inf values, and duplicates.

    Parameters:
    - X_LF (np.ndarray): Low-fidelity data matrix (n_dim1, n_dim2, n_points).
    - X_HF (np.ndarray): High-fidelity data matrix (n_dim1, n_dim2, n_points).

    Returns:
    - Tuple: Preprocessed low- and high-fidelity matrices and a global mask.
    """
    X_LF, X_HF, mask_clean = _clean_data(X_LF, X_HF)
    X_LF, X_HF, mask_uniques = _remove_duplicates(X_LF, X_HF)

    mask_global = np.zeros_like(mask_clean, dtype=bool)
    mask_global[np.where(mask_clean)[0][mask_uniques]] = True

    return X_LF, X_HF, mask_global


def _clean_data(X1: np.ndarray, X2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove snapshots from dataset if NaN and Inf values are present.
    Parameters:
    - X1 (numpy.ndarray): Low-fidelity data matrix of shape (n_dim1, n_dim2, n_points).
    - X2 (numpy.ndarray): High-fidelity data matrix of shape (n_dim1, n_dim2, n_points).

    Returns:
    - Tuple[numpy.ndarray, numpy.ndarray]: Cleaned low- and high-fidelity data matrices
        of dimension (n_dim1, n_dim2, n_points_), where n_points_ <= n_points.
    """
    if X1.ndim != X2.ndim:
        raise ValueError(
            f"Input matrices must have the same shape, got {X1.shape} and {X2.shape}."
        )
    if X1.ndim != 3:
        raise ValueError(f"Input matrices must be 3-dimensional. Got shape {X1.shape}.")

    shape_X1 = X1.shape[:2]
    shape_X2 = X2.shape[:2]
    X1 = _flatten_snapshots(X1)
    X2 = _flatten_snapshots(X2)

    mask_nan1 = np.any(np.isnan(X1), axis=1)
    mask_inf1 = np.any(np.isinf(X1), axis=1)
    mask1 = mask_nan1 | mask_inf1
    mask_nan2 = np.any(np.isnan(X2), axis=1)
    mask_inf2 = np.any(np.isinf(X2), axis=1)
    mask2 = mask_nan2 | mask_inf2
    mask = mask1 | mask2
    mask_clean = ~mask
    X1 = X1[mask_clean, :]
    X2 = X2[mask_clean, :]

    return (
        _unflatten_snapshots(X1, shape_X1),
        _unflatten_snapshots(X2, shape_X2),
        mask_clean,
    )


def _remove_duplicates(
    X_LF: np.ndarray, X_HF: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Check if the data points are unique and remove duplicates.
    Parameters:
    - X_LF (numpy.ndarray): Low-fidelity data matrix of shape (n_dim1, n_dim2, n_points).
    - X_HF (numpy.ndarray): High-fidelity data matrix of shape (n_dim1, n_dim2, n_points).

    Returns:
    - Tuple[numpy.ndarray, numpy.ndarray]: Unique low- and high-fidelity data matrices.
    """
    if X_LF.ndim != X_HF.ndim:
        raise ValueError(
            f"Input matrices must have the same shape, got {X_LF.shape} and {X_HF.shape}."
        )
    if X_LF.ndim != 3:
        raise ValueError(
            f"Input matrices must be 3-dimensional. Got shape {X_LF.shape}."
        )

    shape_X_LF = X_LF.shape[:2]
    shape_X_HF = X_HF.shape[:2]
    n_points = X_LF.shape[2]
    assert n_points == X_HF.shape[2]
    X_LF = _flatten_snapshots(X_LF)
    X_HF = _flatten_snapshots(X_HF)

    _, unique_idxs_LF = np.unique(X_LF, axis=0, return_index=True)
    unique_idxs_LF = np.sort(unique_idxs_LF)
    X_LF, X_HF = X_LF[unique_idxs_LF, :], X_HF[unique_idxs_LF, :]

    mask_uniques = np.zeros(n_points, dtype=bool)
    mask_uniques[unique_idxs_LF] = True

    return (
        _unflatten_snapshots(X_LF, shape_X_LF),
        _unflatten_snapshots(X_HF, shape_X_HF),
        mask_uniques,
    )


def _remove_outliers(X, Y, max_z_score=3, axis=0):
    """
    Remove outliers from the 2-d data based on the z-score of each data point.
    Parameters:
    - X (numpy.ndarray): Reference data matrix.
    - Y (numpy.ndarray): Dependant data matrix.
    - max_z_score (float): Maximum z-score allowed.
    - axis (int): Axis along which to calculate the z-score.
    """
    # Calculate mean and standard deviation along each dimension
    mean = np.mean(X, axis=axis)
    std_dev = np.std(X, axis=axis)

    # Calculate z-score for each data point
    if axis == 0:
        z_scores = np.abs((X - mean) / std_dev)
    else:
        z_scores = np.abs((X.T - mean) / std_dev).T

    # Find data points where any dimension's z-score is greater than 2
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale the data based on the dataset.

    Parameters:
    - X_LF (numpy.ndarray): Low-fidelity data matrix.
    - X_HF (numpy.ndarray): High-fidelity data matrix.
    - dataset_name (str): Name of the dataset.

    Returns:
    - tuple: Scaled low- and high-fidelity data matrix X_LF and X_HF.
    """

    normalizers = {
        "inclusion-field": _normalize_inclusion_field,
        "darcy-flow": _normalize_darcy_flow,
        "inclusion-qoi": _normalize_inclusion_qoi,
    }

    if dataset_name not in normalizers:
        raise ValueError(
            f"Invalid dataset name. Expected one of {normalizers.keys()}, got {dataset_name} instead."
        )

    return normalizers[dataset_name](X_LF, X_HF, return_normalization_vars)


def _normalize_inclusion_field(
    X_LF: np.ndarray, X_HF: np.ndarray, return_normalization_vars: bool
) -> Tuple[np.ndarray, np.ndarray]:

    # Remove mean vertical displacement field UY_mean that goes from 0 on the bottom to the max value on the top
    n_dim = X_LF.shape[1]
    UY_mean = (
        np.ones_like(X_LF)
        * np.linspace(0, np.mean(X_LF[-1, :, :], axis=0), n_dim)[:, np.newaxis, :]
    )
    X_LF -= UY_mean
    X_HF -= UY_mean
    # Scale each snapshot based on its maximum value
    X_LF_scale = np.max(np.abs(X_LF), axis=(0, 1)) + 1e-8
    X_LF /= X_LF_scale
    X_HF /= X_LF_scale

    if return_normalization_vars:
        return X_LF, X_HF, {"X_mean": UY_mean, "X_scale": X_LF_scale}
    return X_LF, X_HF


def _normalize_darcy_flow(
    X_LF: np.ndarray, X_HF: np.ndarray, return_normalization_vars: bool
) -> Tuple[np.ndarray, np.ndarray]:
    X_LF_mean = np.mean(X_LF, axis=(0, 1))
    X_HF_mean = np.mean(X_HF, axis=(0, 1))
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


def _normalize_inclusion_qoi(
    X_LF: np.ndarray, X_HF: np.ndarray, return_normalizations_vars: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize the inclusion QoI dataset.
    Note: This dataset is originally 2-d with shape (n_dim, n_points), but it has been reshaped
    to 3-d (n_dim1, 1, n_points), because all other utility funcitons work with 3-d datasets.
    This is the reason for all the reshaping happaning in this function.
    """

    def _generate_inclusion_qoi(X):
        n_dim1, n_points = X.shape
        snap_dim = 4
        step = n_dim1 // snap_dim
        U = np.zeros((snap_dim + 1, n_points))

        for i in range(snap_dim):
            u = np.trapz(
                X[i * step : (i + 1) * step, :], dx=0.001, axis=0  # noqa: E203
            )  # noqa: E203
            U[i, :] = u
        U[-1, :] = np.max(X, axis=0)
        return U

    X_LF, X_HF = X_LF.squeeze(axis=1), X_HF.squeeze(
        axis=1
    )  # Get rid of additional axis

    # diff = X_LF - np.mean(X_LF, axis=0).reshape(1, -1)
    # X_LF += 0.45 * diff

    X_LF = _generate_inclusion_qoi(X_LF)
    X_HF = _generate_inclusion_qoi(X_HF)

    X_LF, X_HF = _remove_outliers(X_LF, X_HF, max_z_score=3.5, axis=1)

    X_LF_mean, X_LF_std = np.mean(X_LF, axis=1), np.std(X_LF, axis=1)

    X_LF = _normalize(X_LF.T, X_LF_mean, X_LF_std).T
    X_HF = _normalize(X_HF.T, X_LF_mean, X_LF_std).T

    if return_normalizations_vars:
        return (
            X_LF[:, np.newaxis, :],
            X_HF[:, np.newaxis, :],
            {"X_mean": X_LF_mean, "X_scale": X_LF_std},
        )

    return X_LF[:, np.newaxis, :], X_HF[:, np.newaxis, :]  # Reshape back to 3-d


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
    Flatten the input data matrices.

    Parameters:
    - X_LF (numpy.ndarray): Low-fidelity data matrix of shape (n_dim1, n_dim2, n_points).
    - X_HF (numpy.ndarray): High-fidelity data matrix of shape  (n_dim1, n_dim2, n_points).

    Returns:
    - tuple: Flattened low- and high-fidelity data matrices X_LF and X_HF of shape
    (n_points, n_dim1 * n_dim2).
    """
    return _flatten_snapshots(X_LF), _flatten_snapshots(X_HF)


def _flatten_snapshots(X: np.ndarray) -> np.ndarray:
    """
    Flatten the snapshots in the data matrix of shape (n_dim1, n_dim2, n_points)
    into a matrix of shape (n_points, n_dim1 * n_dim2).
    """
    if X.ndim != 3:
        raise ValueError(
            f"To flatten, input matrix must be 3-dimensional. Got shape {X.shape}."
        )
    n_points = X.shape[-1]
    U = np.array([X[:, :, i].flatten() for i in range(n_points)])
    return U


def _unflatten_snapshots(X: np.ndarray, shape_X: tuple = None) -> np.ndarray:
    """
    Unflatten the snapshots in the data matrix of shape (n_points, n_dim1 * n_dim2)
    into a matrix of shape (n_dim1, n_dim2, n_points).
    If shape_X is provided, the unflattened data will have the specified shape.
    Otherwise, it will be assumed that n_dim1 = n_dim2.
    """
    if X.ndim != 2:
        raise ValueError(f"Input matrix must be 2-dimensional, got shape {X.shape}.")

    n_points = X.shape[0]
    if shape_X is None:
        n_dim1 = int(np.sqrt(X.shape[1]))
        n_dim2 = n_dim1
    else:
        n_dim1, n_dim2 = shape_X

    X_ = np.zeros((n_dim1, n_dim2, n_points))
    for i in range(n_points):
        X_[:, :, i] = X[i, :].reshape(n_dim1, n_dim2)
    return X_
