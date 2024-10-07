# This file contains utility functions for graph construction and manipulation.
import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import Callable, Union
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph


class GraphCore:
    DEFAULT_K_ADJ = 7
    DEFAULT_K_NN_RATIO = 0.1
    DEFAULT_P = 0.5
    DEFAULT_Q = 0.5

    def __init__(
        self,
        data: np.ndarray,
        metric: Union[str, Callable],
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
        GraphCore class includes methods to compute representations of a graph, such as adjacency and normalized
        graph Laplacian matrix of a graph.

        Methods:
        - compute_adjacency: Compute the adjacency matrix of the graph.
        - compute_graph_laplacian: Compute the normalized graph Laplacian of the graph

        Parameters:
        - data (numpy.ndarray): The data matrix of shape (n_samples, n_features).
        - metric (str or None): Metric to measure samples similarity. Default is 'euclidean'.
        - dist_space (str): The space where the distances are computed. Options are ['ambient', 'pod'].
            Default is 'ambient'.
        - n_components (int): Number of POD components to compute distance. Required if dist_space is 'pod'.
        - method (str): Method used to compute the adjacency matrix. Can be 'full' or 'k-nn'. Default is 'full'.
        - k_nn (int): Number of nearest neighbors to use if method is 'k-nn'. Default is None.
        - scale (float or None): Correlation scale parameter for adjacency computation.
            If None, adjacency is computed in 'self_tuning' mode.
        - k_adj (int): Parameter for self-tuning adjacency. Default is 7.
        - kernel_fn (Callable): Kernel function to compute adjacency. Default is Gaussian kernel.
        - p (float): Left exponent for graph Laplacian normalization. Default is 0.5.
        - q (float): Right exponent for graph Laplacian normalization. Default is 0.5.
        """

        if data.ndim != 2:
            raise ValueError(
                f"Data matrix must be 2-dimensional, got shape {data.shape}."
            )

        self.data = data
        self.metric = metric
        self.dist_space = dist_space
        self.n_components = n_components
        self.method = method
        self.k_nn = k_nn
        self.corr_scale = corr_scale
        self.k_adj = k_adj
        self.kernel_fn = kernel_fn
        self.p = p
        self.q = q

        self._validate_params()

    @staticmethod
    def _log_warning(message: str):
        """Log a warning message."""
        print(f"UserWarning: {message}")

    @staticmethod
    def _gaussian_kernel(dist_matrix: np.ndarray) -> np.ndarray:
        """Apply Gaussian kernel to distance matrix."""
        return np.exp(-(dist_matrix**2))

    def compute_adjacency(self) -> np.ndarray:
        """
        Computes the adjacency matrix for the graph of the input data.

        Returns:
        - numpy.ndarray: Adjacency matrix.
        """
        # Transform the input data based on dist_space (POD/PCA or ambient)
        transformed_data = self._transform_data()
        # Compute distance matrix based on metric and method
        distance_matrix = self._compute_distance_matrix(transformed_data)
        # Scale distance matrix
        scaled_distance_matrix = self._scale_distance_matrix(distance_matrix)
        # Compute the final adjacency matrix
        adjacency_matrix = self._apply_kernel_fn(scaled_distance_matrix)

        return adjacency_matrix

    def compute_graph_laplacian(self, W: np.ndarray) -> np.ndarray:
        """
        Calculate the normalized graph Laplacian of the graph.

        L = D^p * (D - W) * D^q

        Parameters:
        - W (numpy.ndarray): Adjacency matrix.
        - p (float): Left exponent for normalization.
        - q (float): Right exponent for normalization.

        Returns:
        - numpy.ndarray: Normalized graph Laplacian matrix.
        """
        if self.p is None:
            self.p = self.DEFAULT_P
            self._log_warning(
                f"Normalization exponent 'p' not provided. Using default value of {self.p}."
            )
        if self.q is None:
            self.q = self.DEFAULT_Q
            self._log_warning(
                f"Normalization exponent 'q' not provided. Using default value of {self.q}."
            )
        if not isinstance(self.p, (float, int)) or not isinstance(self.q, (float, int)):
            raise ValueError(
                "Normalization exponents must be of type float or int. "
                f"Got {type(self.p)} and {type(self.q)} instead."
            )

        d = np.sum(W, axis=1)
        D = np.diag(d)

        if self.p == 0 and self.q == 0:
            return D - W

        Dp = np.diag(d ** (-self.p)) if self.p != 0 else None
        Dq = np.diag(d ** (-self.q)) if self.q != 0 else None

        if Dp is None:
            return (D - W) @ Dq
        if Dq is None:
            return Dp @ (D - W)

        return Dp @ (D - W) @ Dq

    def _transform_data(self) -> np.ndarray:
        """
        Transform the input data based on the specified distance space.

        Returns:
        - numpy.ndarray: Transformed data.
        """
        if self.dist_space == "pod":
            return PCA(n_components=self.n_components).fit_transform(self.data)
        elif self.dist_space == "ambient":
            if self.n_components is not None:
                self._log_warning(
                    "When distance is computed in ambient space, variable 'n_components' is ignored."
                )
            return self.data
        else:
            raise ValueError(
                f"Invalid dist_space. Expected one of ['ambient', 'pod']. Got {self.dist_space}."
            )

    def _compute_distance_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        Compute the distance matrix based on the specified method.

        Parameters:
        - data (numpy.ndarray): Input data matrix.

        Returns:
        - numpy.ndarray: Distance matrix.
        """
        if self.method == "full":  # full distance matrix
            if self.k_nn is not None:
                self._log_warning(
                    "When 'method' is 'full', variable 'k_nn' will be ignored."
                )
            return squareform(pdist(data, self.metric))
        elif self.method == "k-nn":  # approximate distance matrix using k-nn
            if self.k_nn is None:
                self.k_nn = int(self.DEFAULT_K_NN_RATIO * data.shape[0])
                self._log_warning(
                    f"Number of neighbors not provided. Using default value of"
                    f"{self.DEFAULT_K_NN_RATIO * 100}% of n_samples = {self.k_nn}."
                )
            return self._knn_dist_matrix(data, self.k_nn, self.metric)
        else:
            raise ValueError(
                f"Invalid method. Expected one of ['full', 'k-nn'], got {self.method}."
            )

    def _knn_dist_matrix(
        self, data: np.ndarray, k: int, metric: Union[str, Callable]
    ) -> np.ndarray:
        """
        Generate adjacency matrix using k-d tree.

        Parameters:
        - data (numpy.ndarray): Input data matrix.
        - k (int): Number of neighbors for adjacency computation.
        - metric (Union[str, Callable]): Distance metric.

        Returns:
        - numpy.ndarray: Adjacency matrix.
        """
        dist_matrix = kneighbors_graph(
            data, k, mode="distance", metric=metric
        ).toarray()
        dist_matrix[dist_matrix == 0] = np.inf
        return dist_matrix

    def _scale_distance_matrix(self, dist_matrix: np.ndarray) -> np.ndarray:
        """
        Scale the distance matrix based on the specified correlation scale.

        Parameters:
        - dist_matrix (numpy.ndarray): Distance matrix.

        Returns:
        - numpy.ndarray: Scaled distance matrix.
        """
        if self.corr_scale is None:  # self-tuning distance scaling
            if (
                self.k_adj is not None
                and self.method == "k-nn"
                and self.k_nn < self.k_adj
            ):
                raise ValueError(
                    "Number of neighbors for self-tuning adjacency 'k_adj' must be less than 'k_nn'."
                )
            return self._self_tuned_scaling(dist_matrix, self.k_adj)
        else:  # fixed scaling
            return dist_matrix / self.corr_scale

    def _self_tuned_scaling(self, dist_matrix: np.ndarray, k: int) -> np.ndarray:
        """
        Self-tuned scaling of distance matrix.

        Parameters:
        - dist_matrix (numpy.ndarray): Pairwise distance matrix of the input data.
        - k (int): correlation scale for each sample is equal to its distance with the k-th neighbor.

        Returns:
        - numpy.ndarray: Self-tuned scaled distance matrix.
        """
        if k is None:
            k = self.DEFAULT_K_ADJ
            self._log_warning(
                f"Number of neighbors not provided. Using default value of k_adj = {k}."
            )
        if k > dist_matrix.shape[0]:
            raise ValueError(
                f"Number of neighbors k must be less than 'n_nodes'. "
                f"Got k = {k} with n_nodes = {dist_matrix.shape[0]}."
            )

        scales = np.sqrt(np.sort(dist_matrix, axis=1)[:, k - 1])
        return ((dist_matrix / scales).T / scales).T

    def _apply_kernel_fn(self, scaled_dist_matrix: np.ndarray) -> np.ndarray:
        """
        Compute the adjacency matrix by applying a kernel to the scaled distance matrix.

        Parameters:
        - scaled_dist_matrix (numpy.ndarray): Scaled distance matrix.

        Returns:
        - numpy.ndarray: Adjacency matrix.
        """
        if self.kernel_fn is None:
            self.kernel_fn = self._gaussian_kernel
        adj = self.kernel_fn(scaled_dist_matrix)
        np.fill_diagonal(adj, 0)  # Enforce diagonal elements to be zero
        return adj

    def _validate_params(self):
        """
        Check the graph configuration.
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
            raise ValueError(
                f"Number of neighbors for 'k-nn' method must be an integer. Got {self.k_nn}."
            )

        if self.corr_scale is not None and not isinstance(
            self.corr_scale, (float, int)
        ):
            raise ValueError(
                f"Scale must be None or a float or int. Got {self.corr_scale}."
            )
        if isinstance(self.corr_scale, (float, int)):
            if self.corr_scale <= 0:
                raise ValueError(
                    f"Scale must be None or a positive float or int. Got {self.corr_scale}."
                )
            if self.k_adj is not None:
                self._log_warning(
                    "When 'corr_scale' is provided, 'k_adj' will be ignored."
                )
        if self.corr_scale is None:
            if not isinstance(self.k_adj, int) or self.k_adj <= 0:
                raise ValueError(f"k_adj must be a positive integer. Got {self.k_adj}.")
