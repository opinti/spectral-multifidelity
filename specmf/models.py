import numpy as np
from specmf.utils import ordered_eig, spectral_clustering
from specmf.graph_core import GraphCore
from functools import cached_property
from scipy.linalg import solve


class Graph(GraphCore):
    def __init__(self, *args, **kwargs) -> None:
        """
        Graph class is a subclass of GraphCore, and stores graph representations and quantities,
        like adjacency matrix, graph Laplacian, and eigenvectors and eigenvalues of the graph
        Laplacian. All computations are performed by methods of GraphCore class.

        Arguments:
        - data (numpy.ndarray): The data matrix of shape (n_samples, n_dim).
        - metric (str or None): Metric to measure samples similarity. Default is 'euclidean'.
        - dist_space (str): The space where the distances are computed. Options are ['ambient', 'pod'].
            Default is 'ambient'.
        - n_components (int): Number of POD components to compute distance. Required if dist_space is 'pod'.
        - method (str): Method used to compute the adjacency matrix. Can be 'full' or 'k-nn'. Default is 'full'.
        - k_nn (int): Number of nearest neighbors to use if method is 'k-nn'. Default is None.
        - scale (float or None): Correlation scale parameter for adjacency computation.
            If None, adjacency is computed in 'self_tuning' mode.
        - k_adj (int): Parameter for self-tuning adjacency. Default is 7.
        - p (float): Left exponent for graph Laplacian normalization. Default is 0.5.
        - q (float): Right exponent for graph Laplacian normalization. Default is 0.5.

        """
        super().__init__(*args, **kwargs)

        self.eigvecs = None
        self.eigvals = None

        self.nodes = kwargs.get("data", None)
        self.n_nodes = self.nodes.shape[0]
        self.n_dim = self.nodes.shape[1]

    @cached_property
    def adjacency(self) -> np.ndarray:
        return self.compute_adjacency()

    @cached_property
    def graph_laplacian(self) -> np.ndarray:
        return self.compute_graph_laplacian(self.adjacency)

    def laplacian_eig(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the eigenvalues and eigenvectors of the graph Laplacian.
        """
        if self.eigvals is None or self.eigvecs is None:
            self.eigvals, self.eigvecs = ordered_eig(self.graph_laplacian)
        return self.eigvals, self.eigvecs

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.nodes[idx, ...]

    def __len__(self) -> int:
        return self.n_nodes


class MultiFidelityModel:
    """
    MultiFidelityModel is a class that performs multi-fidelity modeling using specMF method.

    Parameters:
    - sigma (float): The parameter controlling the noise level in the data. Default is 1e-2.
    - beta (int): The parameter controlling the regularization of the graph Laplacian. Default is 2.
    - kappa (float): The parameter controlling the fidelity of the high-fidelity data. Default is 1e-3.
    - method (str): SpecMF variation to computing multi-fidelity data. Can be 'full' or 'trunc'. Default is 'full'.
    - spectrum_cutoff (bool): Number of eigenvectors used if method is 'trunc'.

    Methods:
    - transform(): Compute multi-fidelity data from low-fidelity graph.
    - cluster(): Perform spectral clustering of the low-fidelity graph.
    - fit(): Fit the hyperparameters of the model.
    - predict(): Predict the high-fidelity version of given low-fidelity data sample.
    - summary(): Print the model configuration.
    """

    REG_EPS = 1e-8
    _contained_params = [
        "sigma",
        "beta",
        "kappa",
        "method",
        "spectrum_cutoff",
        "tau",
        "omega",
        "n_clusters",
        "_is_graph_clustered",
    ]

    def __init__(
        self,
        sigma: float = 1e-2,
        beta: int = 2,
        kappa: float = 1e-3,
        omega: float = None,
        method: str = "full",
        spectrum_cutoff: bool = None,
        tau: float = None,
    ) -> None:

        self.sigma = sigma
        self.beta = beta
        self.kappa = kappa
        self.omega = omega
        self.method = method
        self.spectrum_cutoff = spectrum_cutoff
        self.tau = tau
        self.inds_centroids = None
        self.labels = None
        self.n_clusters = None
        self._is_graph_clustered = False
        self._check_config()

    def transform(
        self, g_LF: Graph, x_HF: np.ndarray, inds_train: list = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Takes low-fidelity graph and high-fidelity data and returns multi-fidelity data.

        Parameters:
        - g_LF (Graph): The low-fidelity graph.
        - x_HF (np.ndarray): The high-fidelity data (n_points_HF, n_dim).
        - inds_train (list): Indices that provide a one-to-one map between the high-fidelity contained
            in x_HF and the low-fidelity data contained in g_LF.nodes.

        Returns:
        - tuple: The computed multi-fidelity data and the uncertainty estimates.
        """
        assert g_LF.nodes.shape[1] == x_HF.shape[1], "Dimension mismatch."

        if inds_train is None and self.ind_centroids is None:
            raise ValueError(
                "Indices that map the high-to-low-fidelity data must be provided."
            )
        elif inds_train is None:
            inds_train = self.inds_centroids

        if len(inds_train) != x_HF.shape[0]:
            raise ValueError(
                "Number of high-fidelity data points does not match the number of indices provided. "
                f"Got n_points_HF = {x_HF.shape[0]} and {len(inds_train)=}."
            )

        eigvals, _ = g_LF.laplacian_eig()

        if self.tau is None:
            # tau is equal to eigenvalue corresponding to spectral gap.
            # TODO: test this part of the code.
            eigvals_ = np.abs(eigvals[:50])
            log_eigvals = np.log10(eigvals_)
            log_curvature = log_eigvals[:-2] + log_eigvals[2:] - 2 * log_eigvals[1:-1]
            self.tau = eigvals[np.argmin(log_curvature) + 1]

        if self.omega is None:
            self.omega = self.kappa / (self.tau**self.beta)

        if self.method == "full":
            return self._compute_specmf_data(g_LF, x_HF, inds_train)
        elif self.method == "trunc":
            return self._compute_specmf_data_trunc(g_LF, x_HF, inds_train)

    def cluster(
        self, g_LF: Graph, n: int, new_clustering: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform spectral clustering of the low-fidelity graph and return the indices of the
        centroids of the clusters and the labels of the data points.

        Parameters:
        - g_LF (Graph): The low-fidelity graph.
        - n (int): Number of clusters.
        - new_clustering (bool): If True, the clusters are always recomputed.

        Returns:
        - tuple: Indices of the centroids of the clusters, labels of the data points.
        """
        if n <= 0:
            raise ValueError("Number of clusters must be positive.")
        if n >= g_LF.n_nodes:
            raise ValueError(
                f"Value of 'n' must be less than than the total number of data. "
                f"Got n_HF = {n} with n_data = {g_LF.n_nodes}."
            )
        if self.n_clusters is not None and n != self.n_clusters:
            self.inds_centroids, self.labels = None, None
            print(
                "UserWarning: the 'cluster' method has already been used with this model "
                "with a different value of 'n'. With a new call the clusters are recomputed "
                " and the attributes inds_centroinds and labels are overwritten."
            )
        if n == self.n_clusters and not new_clustering:
            return self.inds_centroids, self.labels

        self.n_clusters = n
        _, eigvecs = g_LF.laplacian_eig()
        self.inds_centroids, self.labels = spectral_clustering(eigvecs, n)
        self._is_graph_clustered = True
        return self.inds_centroids, self.labels

    def _compute_specmf_data(
        self, g_LF: Graph, x_HF: np.ndarray, inds_train: list
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute multi-fidelity data using standard specMF method.
        """
        x_LF = g_LF.nodes
        L = g_LF.graph_laplacian

        n_LF = x_LF.shape[0]
        n_HF = len(inds_train)
        assert (
            n_HF == x_HF.shape[0]
        ), "Number of high-fidelity data does not match the number of indices."

        Phi_hat = x_HF - x_LF[inds_train, :]
        P_N = np.zeros((n_HF, n_LF))
        P_N[np.arange(n_HF), inds_train] = 1

        L_reg = np.linalg.matrix_power(L + self.tau * np.eye(n_LF), self.beta)

        B = 1 / self.sigma**2 * P_N.T @ P_N + self.omega * L_reg

        # Solve for the inverse of B
        C_phi = solve(B, np.eye(n_LF))
        # C_phi = np.linalg.inv(B)

        # Compute Phi_mean
        Phi_mean = C_phi @ (1 / self.sigma**2 * P_N.T @ Phi_hat)

        x_MF = x_LF + Phi_mean
        dPhi = np.sqrt(np.diag(C_phi) + self.REG_EPS)

        return x_MF, C_phi, dPhi

    def _compute_specmf_data_trunc(
        self, g_LF: Graph, x_HF: np.ndarray, inds_train: list
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute multi-fidelity data using the truncated specMF method.
        """
        assert (
            len(inds_train) == x_HF.shape[0]
        ), "Number of high-fidelity data does not match the number of indices."

        x_LF = g_LF.nodes
        eigvals, eigvecs = g_LF.laplacian_eig()

        Psi = eigvecs[:, : self.spectrum_cutoff]
        Psi_N = Psi[inds_train, :]
        Phi_hat = x_HF - x_LF[inds_train, :]

        eigvals_reg = (np.abs(eigvals[: self.spectrum_cutoff]) + self.tau) ** self.beta

        B = 1 / self.sigma**2 * Psi_N.T @ Psi_N + self.omega * np.diag(eigvals_reg)
        C_a = solve(B, np.eye(self.spectrum_cutoff))
        # C_a = np.linalg.inv(B)
        A = 1 / self.sigma**2 * C_a @ Psi_N.T @ Phi_hat
        Phi_mean = Psi @ A

        x_MF = x_LF + Phi_mean

        C_phi = Psi @ C_a @ Psi.T
        dPhi = np.sqrt(np.diag(C_phi) + self.REG_EPS)

        return x_MF, C_phi, dPhi

    def fit(
        self,
        *args,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit the hyperparameters of the model based on the validation data.
        """
        raise NotImplementedError

    def predict(x_LF: np.ndarray) -> np.ndarray:
        """
        Predict the high-fidelity version of the given low-fidelity data sample.
        """
        raise NotImplementedError

    def summary(self):
        """
        Print the model configuration.
        """
        max_key_length = max(len(key) for key in self.__dict__.keys())

        print(int(4 * max_key_length) * "=")
        print("Model Configuration:")
        print(int(4 * max_key_length) * "=")

        for key, value in self.__dict__.items():
            if key in self._contained_params:
                print(f"{key.ljust(max_key_length + 4)} : {value}")

        print(int(4 * max_key_length) * "=")

    def _check_config(self):
        if self.method not in ["full", "trunc"]:
            raise ValueError(
                f"Invalid method. Expected 'full' or 'trunc', got {self.method}."
            )
        if self.method == "trunc" and self.spectrum_cutoff is None:
            raise ValueError(
                "With method 'trunc', parameter 'spectrum-cutoff' must be provided."
            )
        if self.method == "full" and self.spectrum_cutoff is not None:
            print(
                "Warning: When method is 'full' the parameter 'spectrum_cutoff' is ignored."
            )
        if self.sigma is None or self.sigma <= 0:
            raise ValueError(f"Parameter 'sigma' must be positive, got {self.sigma}.")
        if self.beta is None or self.beta <= 0:
            raise ValueError(f"Parameter 'beta' must be positive, got {self.beta}.")
        if self.omega is None:
            if self.kappa is None or self.kappa <= 0:
                raise ValueError(
                    f"Parameter 'kappa' must be positive, got {self.kappa}."
                )
            if self.spectrum_cutoff is not None and self.spectrum_cutoff <= 0:
                raise ValueError(
                    f"Parameter 'spectrum_cutoff' must be positive, got {self.spectrum_cutoff}."
                )
        else:
            if self.omega <= 0:
                raise ValueError(
                    f"Parameter 'omega' must be positive, got {self.omega}."
                )
            if self.kappa is not None:
                print("Warning: Parameter 'kappa' is ignored when 'omega' is provided.")
        if self.tau is not None and self.tau <= 0:
            raise ValueError(f"Parameter 'tau' must be positive, got {self.tau}.")
