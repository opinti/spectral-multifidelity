import numpy as np
from specmf.utils import ordered_eig, spectral_clustering
from specmf.graph_core import GraphCore
from functools import cached_property
from scipy.linalg import solve


class Graph(GraphCore):
    """
    Graph class is a subclass of GraphCore that stores graph representations
    and quantities like the adjacency matrix, graph Laplacian, and eigenvectors
    and eigenvalues of the graph Laplacian. All computations are performed by
    the GraphCore methods.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eigvecs = None
        self.eigvals = None

        self.nodes = kwargs.get("data")
        self.n_nodes, self.n_dim = self.nodes.shape

    @cached_property
    def adjacency(self) -> np.ndarray:
        """Compute and return the adjacency matrix."""
        return self.compute_adjacency()

    @cached_property
    def graph_laplacian(self) -> np.ndarray:
        """Compute and return the graph Laplacian."""
        return self.compute_graph_laplacian(self.adjacency)

    def laplacian_eig(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the eigenvalues and eigenvectors of the graph Laplacian."""
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
        spectrum_cutoff: int = None,
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
                f"Number of high-fidelity data points does not match the number of indices. "
                f"Got {x_HF.shape[0]} high-fidelity points and {len(inds_train)} indices."
            )

        eigvals, _ = g_LF.laplacian_eig()

        if self.tau is None:
            self.tau = self._compute_spectral_gap(eigvals)

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
        if n <= 0 or n >= g_LF.n_nodes:
            raise ValueError(
                f"Invalid number of clusters: {n}. Must be positive and less than {g_LF.n_nodes})."
            )

        if self.n_clusters and n != self.n_clusters:
            print(
                "UserWarning: Clusters have already been computed with a different 'n', recomputing..."
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
        """Compute multi-fidelity data using standard specMF method."""
        x_LF = g_LF.nodes
        L = g_LF.graph_laplacian

        n_LF, n_HF = x_LF.shape[0], len(inds_train)
        Phi_hat = x_HF - x_LF[inds_train, :]
        P_N = np.zeros((n_HF, n_LF))
        P_N[np.arange(n_HF), inds_train] = 1

        L_reg = np.linalg.matrix_power(L + self.tau * np.eye(n_LF), self.beta)
        B = (1 / self.sigma**2) * (P_N.T @ P_N) + self.omega * L_reg

        C_phi = solve(B, np.eye(n_LF))
        Phi_mean = C_phi @ ((1 / self.sigma**2) * P_N.T @ Phi_hat)

        x_MF = x_LF + Phi_mean
        dPhi = np.sqrt(np.diag(C_phi) + self.REG_EPS)

        return x_MF, C_phi, dPhi

    def _compute_specmf_data_trunc(
        self, g_LF: Graph, x_HF: np.ndarray, inds_train: list
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute multi-fidelity data using truncated specMF method."""
        x_LF = g_LF.nodes
        eigvals, eigvecs = g_LF.laplacian_eig()

        Psi = eigvecs[:, : self.spectrum_cutoff]
        Psi_N = Psi[inds_train, :]
        Phi_hat = x_HF - x_LF[inds_train, :]

        eigvals_reg = (np.abs(eigvals[: self.spectrum_cutoff]) + self.tau) ** self.beta
        B = (1 / self.sigma**2) * (Psi_N.T @ Psi_N) + self.omega * np.diag(eigvals_reg)

        C_a = solve(B, np.eye(self.spectrum_cutoff))
        A = (1 / self.sigma**2) * C_a @ Psi_N.T @ Phi_hat
        Phi_mean = Psi @ A

        x_MF = x_LF + Phi_mean
        C_phi = Psi @ C_a @ Psi.T
        dPhi = np.sqrt(np.diag(C_phi) + self.REG_EPS)

        return x_MF, C_phi, dPhi

    def _compute_spectral_gap(self, eigvals: np.ndarray) -> float:
        """Compute spectral gap for tau estimation."""
        eigvals_ = np.abs(eigvals[:50])
        log_eigvals = np.log10(eigvals_)
        log_curvature = log_eigvals[:-2] + log_eigvals[2:] - 2 * log_eigvals[1:-1]
        return eigvals[np.argmin(log_curvature) + 1]

    def summary(self) -> None:
        """Print the model configuration."""
        max_key_length = max(
            len(key) for key in self.__dict__ if key in self._contained_params
        )
        divider = "=" * 3 * max_key_length

        print(divider)
        print("Model Configuration:")
        print(divider)

        for key, value in self.__dict__.items():
            if key in self._contained_params:
                print(f"{key.ljust(max_key_length + 4)}: {value}")

        print(divider)

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
        if self.spectrum_cutoff is not None and self.spectrum_cutoff <= 0:
            raise ValueError(
                f"Parameter 'spectrum_cutoff' must be positive, got {self.spectrum_cutoff}."
            )
        if self.omega is not None:
            if self.omega <= 0:
                raise ValueError(
                    f"Parameter 'omega' must be positive, got {self.omega}."
                )
            if self.kappa is not None:
                print("Warning: Parameter 'kappa' is ignored when 'omega' is provided.")
        if self.omega is None and (self.kappa is None or self.kappa <= 0):
            raise ValueError(f"Parameter 'kappa' must be positive, got {self.kappa}.")
        if self.tau is not None and self.tau <= 0:
            raise ValueError(f"Parameter 'tau' must be positive, got {self.tau}.")
