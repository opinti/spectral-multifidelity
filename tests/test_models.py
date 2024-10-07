import numpy as np
import pytest
from specmf.models import MultiFidelityModel, Graph


def test_graph_creation():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    config = {'k': 2}
    graph = Graph(X, **config)
    assert graph.n_nodes == 3
    assert graph.n_dim == 2
    assert np.array_equal(graph.nodes, X)


def test_adjacency_matrix():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    config = {
        'metric': 'sqeuclidean',
        'mode': 'fixed_scale',
        'scale': 0.5,
        'k': 2
    }
    graph = Graph(X, **config)
    adjacency = graph.adjacency
    expected_dist = np.array([[0., 1, 1, 2],
                              [1, 0, 2, 1],
                              [1, 2, 0, 1],
                              [2, 1, 1, 0]])
    expected_adjacency = np.exp(-expected_dist ** 2 / config['scale'] ** 2)
    np.fill_diagonal(expected_adjacency, 0)
    assert np.allclose(adjacency, expected_adjacency)


def test_graph_laplacian():
    # TODO: Implement test to check if graph_laplacian is correctly computed
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    config = {
        'metric': 'euclidean',
        'mode': 'fixed_scale',
        'scale': 0.5,
        'k': 2
    }
    graph = Graph(X, **config)
    laplacian = graph.graph_laplacian
    assert laplacian.shape == (4, 4)  # Checking if the shape is correct
    assert np.allclose(laplacian, laplacian.T)  # Checking if the matrix is symmetric


def test_getitem():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    config = {'k': 2}
    graph = Graph(X, **config)
    assert np.array_equal(graph[0], np.array([1, 2]))
    assert np.array_equal(graph[1], np.array([3, 4]))
    assert np.array_equal(graph[2], np.array([5, 6]))
    with pytest.raises(IndexError):
        graph[3]


CONFIG_ADJ_ERROR = {
    "test-2": {
        # Test case 2: Invalid distance function
        'metric': 'invalid',
        'mode': 'self_tuning',
        'scale': 0.5
    },
    "test-3": {
        # Test case 3: Invalid mode
        'metric': 'euclidean',
        'mode': 'invalid',
        'scale': 0.5
    },
    "test-4": {
        # Test case 4: Missing scale parameter in fixed_scale mode
        'metric': 'euclidean',
        'mode': 'fixed_scale'
    },
    "test-5": {
        # Test case 5: Negative scale parameter
        'metric': 'euclidean',
        'mode': 'fixed_scale',
        'scale': -0.5
    },
    "test-6": {
        # Test case 6: Scale parameter as array
        'metric': 'euclidean',
        'mode': 'fixed_scale',
        'scale': np.array([0.5, 0.5])
    },
    "test-7": {
        # Test case 7: Scale parameter as non-float type
        'metric': 'euclidean',
        'mode': 'fixed_scale',
        'scale': '0.5'
    },
    "test-8": {
        # Test case 8: Missing scale parameter in fixed_scale mode
        'mode': 'fixed_scale',
        'scale': None,
    },
    "test-9": {
        # Test case 9: k parameter too large
        'k': 100,
    },
}


@pytest.mark.parametrize("config_key", CONFIG_ADJ_ERROR.keys())
def test_adjacency_config_errors(config_key):
    X = np.arange(100).reshape(25, 4)
    config_dict = CONFIG_ADJ_ERROR[config_key]
    with pytest.raises(ValueError):
        graph = Graph(X, **config_dict)
        _ = graph.adjacency


CONFIG_ADJ = {
    "test-1": {
        # Test case 1: int scale
        'mode': 'fixed_scale',
        'scale': int(1),
    },
    "test-2": {
        # Test case 3: self-tuning mode
        'mode': 'self_tuning',
    },
    "test-3": {
        # Test case 4: All default parameters
    },
}


@pytest.mark.parametrize("config_key", CONFIG_ADJ.keys())
def test_adjacency_config(config_key):
    X = np.arange(100).reshape(25, 4)
    config_dict = CONFIG_ADJ[config_key]
    graph = Graph(X, **config_dict)
    _ = graph.adjacency


CONFIG_GL = {
    "test-1": {
        # Test case 1: Missing left normalization exponents
        'p': None,
        'q': 0.5,
    },
    "test-2": {
        # Test case 2: Missing right normalization exponents
        'p': 0.5,
        'q': None,
    },
    "test-3": {
        # Test case 3: Missing normalization exponents
        'p': None,
        'q': None,
    },
}


@pytest.mark.parametrize("config_key", CONFIG_GL.keys())
def test_graph_laplacian_config(config_key):
    X = np.arange(100).reshape(25, 4)
    config_dict = CONFIG_GL[config_key]
    graph = Graph(X, **config_dict)
    _ = graph.graph_laplacian


def test_multi_fidelity_model_fit():
    # Create a low-fidelity graph
    X_LF = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    config_LF = {'k': 2}
    graph_LF = Graph(X_LF, **config_LF)

    # Create high-fidelity data
    X_HF = np.array([[2, 2], [2, 3], [3, 2], [3, 3]])

    # Create a multi-fidelity model
    model = MultiFidelityModel(sigma=1e-2, beta=2, kappa=1e-3, method='full')

    # Fit the model
    n_HF = 2
    x_MF, dPhi = model.fit(graph_LF, X_HF, n_HF)

    # Check the shape of the output
    assert x_MF.shape == (4, 2)
    assert dPhi.shape == (4,)


def test_multi_fidelity_model_summary(capsys):
    # Create a multi-fidelity model
    model = MultiFidelityModel(sigma=1e-2, beta=2, kappa=1e-3, method='full')

    # Print the model summary
    model.summary()

    # Capture the printed output
    captured = capsys.readouterr()

    # Check if the summary is printed correctly
    assert "Model Configuration:" in captured.out
    assert "sigma" in captured.out
    assert "beta" in captured.out
    assert "kappa" in captured.out
    assert "method" in captured.out


def test_multi_fidelity_model_invalid_method():
    # Create a multi-fidelity model with an invalid method
    with pytest.raises(ValueError):
        MultiFidelityModel(sigma=1e-2, beta=2, kappa=1e-3, method='invalid')


def test_multi_fidelity_model_invalid_sigma():
    with pytest.raises(ValueError):
        MultiFidelityModel(sigma=-1e-2, beta=2, kappa=1e-3, method='full')


def test_multi_fidelity_model_invalid_beta():
    with pytest.raises(ValueError):
        MultiFidelityModel(sigma=1e-2, beta=-2, kappa=1e-3, method='full')


def test_multi_fidelity_model_invalid_kappa():
    # Create a multi-fidelity model with an invalid kappa
    with pytest.raises(ValueError):
        MultiFidelityModel(sigma=1e-2, beta=2, kappa=-1e-3, method='full')


def test_multi_fidelity_model_invalid_spectrum_cutoff():
    # Create a multi-fidelity model with an invalid spectrum_cutoff
    with pytest.raises(ValueError):
        MultiFidelityModel(sigma=1e-2, beta=2, kappa=1e-3, method='trunc', spectrum_cutoff=0)


if __name__ == '__main__':
    pytest.main()
