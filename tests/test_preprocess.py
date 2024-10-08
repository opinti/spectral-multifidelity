import numpy as np
import pytest
from numpy.testing import assert_array_equal

# Import the functions to be tested
from specmf.preprocess import (
    preprocess_data,
    _clean_data,
    _remove_duplicates,
    _flatten_snapshots,
    _unflatten_snapshots,
)


# Sample test data
@pytest.fixture
def sample_data():
    X_LF = np.arange(3 * 3 * 5).reshape(3, 3, 5).astype(float)
    X_HF = X_LF + 1.0
    return X_LF, X_HF


def test_clean_data(sample_data):
    X_LF, X_HF = sample_data

    # Add NaN and Inf values
    X_LF[0, 0, 0] = np.nan
    X_HF[1, 1, 1] = np.inf

    X_LF_clean, X_HF_clean, mask_clean = _clean_data(X_LF, X_HF)

    assert X_LF_clean.shape == (3, 3, 3)
    assert X_HF_clean.shape == (3, 3, 3)
    assert np.all(mask_clean == np.array([False, False, True, True, True]))


def test_remove_duplicates(sample_data):
    X_LF, X_HF = sample_data

    # Duplicate the last snapshot
    X_LF[:, :, 1] = X_LF[:, :, 0]

    X_LF_unique, X_HF_unique, mask_uniques = _remove_duplicates(X_LF, X_HF)

    assert X_LF_unique.shape == (3, 3, 4)
    assert X_HF_unique.shape == (3, 3, 4)
    assert np.all(mask_uniques == np.array([True, False, True, True, True]))


def test_preprocess_data(sample_data):
    X_LF, X_HF = sample_data

    # Add NaN and Inf values
    X_LF[0, 0, 0] = np.nan
    X_HF[1, 1, 1] = np.inf
    X_LF[:, :, 4] = X_LF[:, :, 3]

    X_LF_processed, X_HF_processed, mask_global = preprocess_data(X_LF, X_HF)

    assert X_LF_processed.shape == (3, 3, 2)
    assert X_HF_processed.shape == (3, 3, 2)
    assert np.all(mask_global == np.array([False, False, True, True, False]))


def test_flatten_snapshots(sample_data):
    X_LF, _ = sample_data

    X_flat = _flatten_snapshots(X_LF)

    assert X_flat.shape == (5, 9)  # Should flatten the snapshots
    with pytest.raises(ValueError):
        _flatten_snapshots(X_LF[0, :, :])


def test_flatten_and_unflatten(sample_data):
    X_LF, _ = sample_data

    X_flat = _flatten_snapshots(X_LF)
    X_unflat = _unflatten_snapshots(X_flat, shape_X=(3, 3))

    assert_array_equal(X_LF, X_unflat)  # Should reconstruct the original array


if __name__ == "__main__":
    import pytest

    pytest.main()
