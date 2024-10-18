import numpy as np
import pytest
from specmf.models import Graph, MultiFidelityModel


@pytest.fixture
def sample_graph_data():
    """Fixture to provide sample data for graph initialization."""
    return np.arange(20).reshape(10, 2).astype(float)


@pytest.fixture
def graph_instance(sample_graph_data):
    """Fixture to initialize a Graph instance."""
    return Graph(data=sample_graph_data)


@pytest.fixture
def sample_high_fidelity_data():
    """Fixture to provide high-fidelity data."""
    return np.arange(20).reshape(10, 2).astype(float) + 1.0


@pytest.fixture
def model_instance():
    """Fixture to initialize MultiFidelityModel instance."""
    return MultiFidelityModel()


# Tests for the Graph class


def test_graph_init(graph_instance):
    """Test if Graph is initialized correctly."""
    graph = graph_instance
    assert graph.n_nodes == 10
    assert graph.n_features == 2
    assert isinstance(graph.adjacency, np.ndarray)
    assert isinstance(graph.graph_laplacian, np.ndarray)
    assert graph.adjacency.shape == (10, 10)


def test_graph_laplacian(graph_instance):
    """Test if graph Laplacian is computed correctly."""
    graph = graph_instance
    laplacian = graph.graph_laplacian
    assert laplacian.shape == (10, 10)


def test_graph_laplacian_eig(graph_instance):
    """Test if Laplacian eigenvalues and eigenvectors are computed correctly."""
    graph = graph_instance
    eigvals, eigvecs = graph.laplacian_eig()
    assert eigvals.shape == (10,)
    assert eigvecs.shape == (10, 10)


def test_graph_getitem(graph_instance):
    """Test __getitem__ method."""
    graph = graph_instance
    node = graph[0]
    assert node.shape == (2,)
    assert np.array_equal(node, graph.nodes[0])


def test_graph_len(graph_instance):
    """Test __len__ method."""
    graph = graph_instance
    assert len(graph) == 10


# Tests for the MultiFidelityModel class


def test_model_init(model_instance):
    """Test if MultiFidelityModel is initialized correctly."""
    model = model_instance
    assert model.sigma == 1e-2
    assert model.beta == 2
    assert model.kappa == 1e-3


def test_model_transform_dimension_mismatch(
    graph_instance, sample_high_fidelity_data, model_instance
):
    """Test if transform method raises an assertion error for dimension mismatch."""
    graph = graph_instance
    model = model_instance
    high_fidelity_data = sample_high_fidelity_data[:, :1]  # Mismatch in dimension

    with pytest.raises(AssertionError, match="Dimension mismatch"):
        model.transform(graph, high_fidelity_data)


def test_model_transform_no_inds_train(
    graph_instance, sample_high_fidelity_data, model_instance
):
    """Test if transform raises an error when no inds_train or centroids are provided."""
    graph = graph_instance
    model = model_instance

    with pytest.raises(
        ValueError,
        match="Indices that map the high-to-low-fidelity data must be provided",
    ):
        model.transform(graph, sample_high_fidelity_data)


def test_model_transform_inds_train_mismatch(
    graph_instance, sample_high_fidelity_data, model_instance
):
    """Test if transform raises an error for inds_train and high-fidelity data mismatch."""
    graph = graph_instance
    model = model_instance

    inds_train = [0, 1]  # Mismatch: two indices, but 10 HF points

    with pytest.raises(
        ValueError,
        match="Number of high-fidelity data points does not match the number of indices",
    ):
        model.transform(graph, sample_high_fidelity_data, inds_train)


def test_model_transform(graph_instance, sample_high_fidelity_data, model_instance):
    """Test if transform method works correctly with valid inputs."""
    graph = graph_instance
    model = model_instance
    inds_train = [0, 1, 2]
    high_fidelity_data_train = sample_high_fidelity_data[inds_train]

    x_MF, C_phi, dPhi = model.transform(graph, high_fidelity_data_train, inds_train)

    assert x_MF.shape == (10, 2)  # Multi-fidelity data should match input dimensions
    assert C_phi.shape == (10, 10)
    assert dPhi.shape == (10,)


def test_model_cluster_invalid_n_clusters(graph_instance, model_instance):
    """Test if cluster method raises an error for invalid number of clusters."""
    graph = graph_instance
    model = model_instance

    with pytest.raises(ValueError, match="Invalid number of clusters"):
        model.cluster(graph, 0)  # Invalid number of clusters


def test_model_cluster(graph_instance, model_instance):
    """Test if clustering works with valid inputs."""
    graph = graph_instance
    model = model_instance

    inds_centroids, labels = model.cluster(graph, n=2)

    assert inds_centroids.shape == (2,)
    assert labels.shape == (10,)
    assert len(set(labels)) == 2  # Should form 2 clusters


if __name__ == "__main__":
    import pytest

    pytest.main()
