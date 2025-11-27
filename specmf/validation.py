"""Input validation utilities for the specmf package."""

import numpy as np


def validate_array_shape(
    arr: np.ndarray, expected_ndim: int, name: str = "array"
) -> None:
    """Validate that an array has expected number of dimensions.

    Parameters
    ----------
    arr : np.ndarray
        Input array to validate.
    expected_ndim : int
        Expected number of dimensions.
    name : str, optional
        Name of the array for error messages. Default is 'array'.

    Raises
    ------
    ValueError
        If array doesn't have the expected dimensions.
    """
    if arr.ndim != expected_ndim:
        raise ValueError(
            f"{name} must be {expected_ndim}D, got shape {arr.shape} instead."
        )


def validate_positive_scalar(value: int | float, name: str = "value") -> None:
    """Validate that a value is a positive scalar.

    Parameters
    ----------
    value : int or float
        Value to validate.
    name : str, optional
        Name of the value for error messages. Default is 'value'.

    Raises
    ------
    ValueError
        If value is not positive.
    """
    if value is None or value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def validate_array_compatibility(
    arr1: np.ndarray,
    arr2: np.ndarray,
    axis: int = 1,
    name1: str = "array1",
    name2: str = "array2",
) -> None:
    """Validate that two arrays are compatible along an axis.

    Parameters
    ----------
    arr1 : np.ndarray
        First array.
    arr2 : np.ndarray
        Second array.
    axis : int, optional
        Axis along which to check compatibility. Default is 1.
    name1 : str, optional
        Name of first array for error messages. Default is 'array1'.
    name2 : str, optional
        Name of second array for error messages. Default is 'array2'.

    Raises
    ------
    ValueError
        If arrays are not compatible.
    """
    if arr1.shape[axis] != arr2.shape[axis]:
        raise ValueError(
            f"{name1} and {name2} must have the same size along "
            f"axis {axis}. Got {arr1.shape[axis]} and "
            f"{arr2.shape[axis]}."
        )


def validate_indices(
    indices: list[int], max_index: int, name: str = "indices"
) -> None:
    """Validate that all indices are within valid range.

    Parameters
    ----------
    indices : list of int
        List of indices to validate.
    max_index : int
        Maximum valid index (exclusive).
    name : str, optional
        Name of the indices for error messages. Default is 'indices'.

    Raises
    ------
    ValueError
        If any index is out of range.
    """
    if not all(0 <= idx < max_index for idx in indices):
        raise ValueError(
            f"All {name} must be in range [0, {max_index}). Got {indices}."
        )


def validate_method_choice(
    method: str, valid_methods: list[str], name: str = "method"
) -> None:
    """Validate that a method choice is valid.

    Parameters
    ----------
    method : str
        Method to validate.
    valid_methods : list of str
        List of valid method names.
    name : str, optional
        Name of the parameter for error messages. Default is 'method'.

    Raises
    ------
    ValueError
        If method is not valid.
    """
    if method not in valid_methods:
        raise ValueError(
            f"Invalid {name}. Expected one of {valid_methods}, got '{method}'."
        )
