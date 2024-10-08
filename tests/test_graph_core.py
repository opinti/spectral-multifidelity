import numpy as np
import pytest
from specmf.graph_core import GraphCore, InvalidMethodError
import logging


@pytest.fixture
def sample_data():
    # Fixture for a sample 2D dataset
    return np.arange(20).reshape(10, 2).astype(float)


def test_graphcore_init_invalid_data_shape():
    # Test if ValueError is raised for invalid data shape
    with pytest.raises(ValueError, match="Data matrix must be 2D"):
        GraphCore(np.array([1, 2, 3]))


def test_graphcore_init_invalid_pod(sample_data):
    # Test if ValueError is raised when 'dist_space' is 'pod' but 'n_components' is not provided
    with pytest.raises(
        ValueError,
        match="Parameter 'n_components' must be provided if 'dist_space' is 'pod'",
    ):
        GraphCore(sample_data, dist_space="pod")


def test_graphcore_init_invalid_n_components_type(sample_data):
    # Test if ValueError is raised for invalid 'n_components' type
    with pytest.raises(ValueError, match="Number of components must be an integer"):
        GraphCore(sample_data, n_components="invalid")


def test_graphcore_init_invalid_k_nn_type(sample_data):
    # Test if ValueError is raised for invalid 'k_nn' type
    with pytest.raises(ValueError, match="'k_nn' must be an integer"):
        GraphCore(sample_data, k_nn="invalid")


def test_graphcore_init_invalid_corr_scale(sample_data):
    # Test if ValueError is raised for invalid 'corr_scale'
    with pytest.raises(ValueError, match="Scale must be a positive float or int"):
        GraphCore(sample_data, corr_scale=-1)


def test_graphcore_init_invalid_k_adj(sample_data):
    # Test if ValueError is raised for invalid 'k_adj'
    with pytest.raises(ValueError, match="'k_adj' must be a positive integer"):
        GraphCore(sample_data, k_adj=-1)


def test_graphcore_init_invalid_kernel_fn(sample_data):
    # Test if ValueError is raised for invalid kernel function (non-callable)
    with pytest.raises(ValueError, match="Kernel function must be callable"):
        GraphCore(sample_data, kernel_fn="not_callable")


def test_graphcore_invalid_method(sample_data):
    # Test if InvalidMethodError is raised for invalid method
    with pytest.raises(
        InvalidMethodError, match="Invalid method 'invalid'. Expected 'full' or 'k-nn'"
    ):
        gc = GraphCore(sample_data, method="invalid")
        gc.compute_adjacency()


def test_graphcore_full_method_ignores_k_nn(sample_data, caplog):
    # Test if a warning is logged when 'k_nn' is provided for 'full' method
    gc = GraphCore(sample_data, method="full", k_nn=8)

    with caplog.at_level(logging.WARNING):
        gc.compute_adjacency()

    assert "'k_nn' is ignored when 'method' is 'full'" in caplog.text


def test_graphcore_knn_default_k_nn(sample_data):
    # Test if default 'k_nn' is set correctly when not provided for 'k-nn' method
    sample_data = np.arange(200).reshape(100, 2).astype(float)
    gc = GraphCore(sample_data, method="k-nn")
    gc.compute_adjacency()
    assert gc.k_nn == int(gc.DEFAULT_K_NN_RATIO * sample_data.shape[0])


def test_graphcore_knn_dist_matrix(sample_data):
    # Test k-NN distance matrix computation
    gc = GraphCore(sample_data, method="k-nn", k_nn=2)
    dist_matrix = gc._knn_dist_matrix(sample_data, 2, gc.metric)
    assert dist_matrix.shape == (10, 10)
    assert np.all(dist_matrix >= 0), "Distance matrix should have non-negative values."


def test_graphcore_adjacency_matrix(sample_data):
    # Test adjacency matrix computation
    gc = GraphCore(sample_data)
    adj_matrix = gc.compute_adjacency()
    assert adj_matrix.shape == (10, 10)
    assert np.allclose(
        np.diag(adj_matrix), 0
    ), "Adjacency matrix diagonal should be zero."

    gc = GraphCore(sample_data, method="k-nn", k_nn=8)
    adj_matrix = gc.compute_adjacency()
    assert adj_matrix.shape == (10, 10)
    assert np.allclose(
        np.diag(adj_matrix), 0
    ), "Adjacency matrix diagonal should be zero."


def test_graphcore_graph_laplacian(sample_data):
    # Test graph Laplacian computation
    gc = GraphCore(sample_data)
    adj_matrix = gc.compute_adjacency()
    laplacian = gc.compute_graph_laplacian(adj_matrix)
    assert laplacian.shape == (10, 10)


def test_graphcore_self_tuned_scaling(sample_data):
    # Test self-tuned scaling function
    gc = GraphCore(sample_data, method="k-nn", k_adj=2)
    dist_matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    scaled_matrix = gc._self_tuned_scaling(dist_matrix, 2)
    assert (
        scaled_matrix.shape == dist_matrix.shape
    ), "Scaled matrix should have the same shape as the distance matrix."


def test_graphcore_self_tuned_scaling_invalid_k(sample_data):
    # Test if ValueError is raised for invalid 'k_adj' in self-tuned scaling
    gc = GraphCore(
        sample_data, method="k-nn", k_adj=10
    )  # k_adj larger than number of samples
    dist_matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    with pytest.raises(
        ValueError, match="'k_adj' must be less than the number of samples"
    ):
        gc._self_tuned_scaling(dist_matrix, 10)


if __name__ == "__main__":
    import pytest

    pytest.main()
