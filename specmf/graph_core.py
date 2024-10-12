import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from typing import Callable, Union
import logging

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class InvalidMethodError(ValueError):
    """Custom exception for invalid method selection."""

    pass


class GraphCore:
    DEFAULT_K_ADJ = 7
    DEFAULT_K_NN_RATIO = 0.1
    DEFAULT_P = 0.5
    DEFAULT_Q = 0.5

    def __init__(
        self,
        data: np.ndarray,
        metric: Union[str, Callable] = "euclidean",
        dist_space: str = "ambient",
        n_components: int = None,
        method: str = "full",
        k_nn: int = None,
        corr_scale: float = None,
        k_adj: int = None,
        kernel_fn: Callable = None,
        p: float = None,
        q: float = None,
    ):
        """
        GraphCore class includes methods to compute representations of a graph,
        such as the adjacency matrix and normalized graph Laplacian.

        Parameters:
        - data (numpy.ndarray): The data matrix of shape (n_samples, n_features).
        - metric (Union[str, Callable]): Metric to measure sample similarity. Default is 'euclidean'.
        - dist_space (str): Space where distances are computed. Options are ['ambient', 'pod']. Default is 'ambient'.
        - n_components (int): Number of components for POD. Required if dist_space is 'pod'.
        - method (str): Method used to compute the adjacency matrix ('full' or 'k-nn'). Default is 'full'.
        - k_nn (int): Number of nearest neighbors for 'k-nn' method. Default is None.
        - corr_scale (float or None): Scale parameter for adjacency computation.
        - k_adj (int): Self-tuning adjacency parameter. Default is 7.
        - kernel_fn (Callable): Kernel function for adjacency computation. Default is Gaussian.
        - p (float): Left exponent for graph Laplacian normalization. Default is 0.5.
        - q (float): Right exponent for graph Laplacian normalization. Default is 0.5.
        """
        if data.ndim != 2:
            raise ValueError(f"Data matrix must be 2D, got shape {data.shape}.")

        self.data = data
        self.metric = metric
        self.dist_space = dist_space
        self.n_components = n_components
        self.method = method
        self.k_nn = k_nn
        self.corr_scale = corr_scale
        self.k_adj = k_adj if k_adj is not None else self.DEFAULT_K_ADJ
        self.kernel_fn = kernel_fn if kernel_fn is not None else self._gaussian_kernel
        self.p = p if p is not None else self.DEFAULT_P
        self.q = q if q is not None else self.DEFAULT_Q

        self._validate_params()

    @staticmethod
    def _log_warning(message: str):
        """Log a warning message."""
        logger.warning(message)

    @staticmethod
    def _gaussian_kernel(dist_matrix: np.ndarray) -> np.ndarray:
        """Apply Gaussian kernel to distance matrix."""
        return np.exp(-(dist_matrix**2))

    def compute_adjacency(self) -> np.ndarray:
        """
        Compute the adjacency matrix for the graph.

        Returns:
        - np.ndarray: The adjacency matrix.
        """
        transformed_data = self._transform_data()
        distance_matrix = self._compute_distance_matrix(transformed_data)
        scaled_distance_matrix = self._scale_distance_matrix(distance_matrix)
        adjacency_matrix = self._apply_kernel_fn(scaled_distance_matrix)

        return adjacency_matrix

    def compute_graph_laplacian(self, W: np.ndarray) -> np.ndarray:
        """
        Compute the normalized graph Laplacian matrix.

        L = D^p * (D - W) * D^q

        Parameters:
        - W (np.ndarray): Adjacency matrix.

        Returns:
        - np.ndarray: The normalized graph Laplacian.
        """
        if not isinstance(self.p, (float, int)) or not isinstance(self.q, (float, int)):
            raise ValueError(
                "Normalization exponents 'p' and 'q' must be float or int. "
                f"Got {type(self.p)} and {type(self.q)} instead."
            )

        d = np.sum(W, axis=1)
        D = np.diag(d)

        if self.p == 0 and self.q == 0:
            return D - W

        D_p = np.diag(d ** (-self.p)) if self.p != 0 else None
        D_q = np.diag(d ** (-self.q)) if self.q != 0 else None

        if D_p is None:
            return (D - W) @ D_q
        if D_q is None:
            return D_p @ (D - W)

        return D_p @ (D - W) @ D_q

    def _transform_data(self) -> np.ndarray:
        """

        Transform input data based on the specified distance space.

        Returns:
        - np.ndarray: Transformed data.
        """
        if self.dist_space == "pod":
            return PCA(n_components=self.n_components).fit_transform(self.data)

        if self.dist_space == "ambient" and self.n_components is not None:
            self._log_warning("In 'ambient' space, 'n_components' is ignored.")

        return self.data

    def _compute_distance_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        Compute the distance matrix based on the specified method.

        Returns:
        - np.ndarray: The distance matrix.
        """
        if self.method == "full":
            if self.k_nn is not None:
                self._log_warning("'k_nn' is ignored when 'method' is 'full'.")
            return squareform(pdist(data, self.metric))

        if self.method == "k-nn":
            if self.k_nn is None:
                self.k_nn = int(self.DEFAULT_K_NN_RATIO * data.shape[0])
                self._log_warning(
                    f"'k_nn' not provided. Using default k_nn = {self.k_nn}."
                )
            return self._knn_dist_matrix(data, self.k_nn, self.metric)

        raise InvalidMethodError(
            f"Invalid method '{self.method}'. Expected 'full' or 'k-nn'."
        )

    def _knn_dist_matrix(
        self, data: np.ndarray, k: int, metric: Union[str, Callable]
    ) -> np.ndarray:
        """
        Generate adjacency matrix using k-nearest neighbors.

        Parameters:
        - data (np.ndarray): Input data matrix.
        - k (int): Number of nearest neighbors.
        - metric (Union[str, Callable]): Distance metric.

        Returns:
        - np.ndarray: The adjacency matrix.
        """
        dist_matrix = kneighbors_graph(data, k, mode="distance", metric=metric)
        dist_matrix = dist_matrix.toarray()  # Convert to dense matrix
        dist_matrix[dist_matrix == 0] = np.inf  # Set 0s (non-neighbors) to infinity.
        return dist_matrix

    def _scale_distance_matrix(self, dist_matrix: np.ndarray) -> np.ndarray:
        """
        Scale the distance matrix based on the specified correlation scale.

        Parameters:
        - dist_matrix (np.ndarray): Distance matrix.

        Returns:
        - np.ndarray: Scaled distance matrix.
        """
        if self.corr_scale is None:
            if self.method == "k-nn" and self.k_nn < self.k_adj:
                raise ValueError(
                    "Number of neighbors for self-tuning adjacency 'k_adj' must be less than 'k_nn'."
                )
            return self._self_tuned_scaling(dist_matrix, self.k_adj)
        else:
            return dist_matrix / self.corr_scale

    def _self_tuned_scaling(self, dist_matrix: np.ndarray, k: int) -> np.ndarray:
        """
        Apply self-tuned scaling to the distance matrix.

        Parameters:
        - dist_matrix (np.ndarray): Distance matrix.
        - k (int): The k-th neighbor to determine the scale for each sample.

        Returns:
        - np.ndarray: Self-tuned scaled distance matrix.
        """
        if k > dist_matrix.shape[0]:
            raise ValueError(
                f"'k_adj' must be less than the number of samples. Got k = {k}."
            )

        scales = np.sqrt(np.sort(dist_matrix, axis=1)[:, k - 1])
        return ((dist_matrix / scales).T / scales).T

    def _apply_kernel_fn(self, scaled_dist_matrix: np.ndarray) -> np.ndarray:
        """
        Apply the kernel function to the scaled distance matrix to compute the adjacency matrix.

        Parameters:
        - scaled_dist_matrix (np.ndarray): Scaled distance matrix.

        Returns:
        - np.ndarray: Adjacency matrix.
        """
        adj = self.kernel_fn(scaled_dist_matrix)
        np.fill_diagonal(adj, 0)  # Enforce zero diagonal for adjacency matrix
        return adj

    def _validate_params(self):
        """
        Validate the input parameters.
        """
        if self.dist_space == "pod" and self.n_components is None:
            raise ValueError(
                "Parameter 'n_components' must be provided if 'dist_space' is 'pod'."
            )

        if self.n_components is not None and not isinstance(self.n_components, int):
            raise ValueError(
                f"Number of components must be an integer. Got {self.n_components} instead."
            )

        if self.k_nn is not None and not isinstance(self.k_nn, int):
            raise ValueError(f"'k_nn' must be an integer. Got {self.k_nn}.")

        if self.corr_scale is not None:
            if not isinstance(self.corr_scale, (float, int)) or self.corr_scale <= 0:
                raise ValueError(
                    f"Scale must be a positive float or int. Got {self.corr_scale}."
                )
            if self.k_adj is not None:
                self._log_warning(
                    "When 'corr_scale' is provided, 'k_adj' will be ignored."
                )

        if not isinstance(self.k_adj, int) or self.k_adj <= 0:
            raise ValueError(f"'k_adj' must be a positive integer. Got {self.k_adj}.")

        if not callable(self.kernel_fn):
            raise ValueError("Kernel function must be callable.")
