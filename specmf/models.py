import logging
from functools import cached_property

import numpy as np
from scipy.linalg import solve

from specmf.graph_core import GraphCore
from specmf.utils import ordered_eig, spectral_clustering
from specmf.validation import (
    validate_array_compatibility,
    validate_array_shape,
    validate_method_choice,
    validate_positive_scalar,
)


# Setup logging
logger = logging.getLogger(__name__)


class Graph(GraphCore):
    """
    Graph class is a subclass of GraphCore that stores graph representations
    and quantities like the adjacency matrix, graph Laplacian, and eigenvectors
    and eigenvalues of the graph Laplacian. All computations are performed by
    the GraphCore methods.

    Methods:
    - adjacency(): Compute and return the adjacency matrix. Cached property.
    - graph_laplacian(): Compute and return the graph Laplacian.
    It is a cached property.
    - laplacian_eig(): Return the eigenvalues and eigenvectors of the
    graph Laplacian. It is a cached property.
    - cluster(): Perform spectral clustering of the graph nodes and return the
    indices of the centroids of the clusters and the labels of the nodes.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eigvecs = None
        self.eigvals = None

        self.nodes = kwargs.get("data")
        self.n_nodes, self.n_features = self.nodes.shape

        self.inds_centroids = None
        self.labels = None
        self.n_clusters = None
        self._is_graph_clustered = False

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

    def cluster(
        self, n: int, new_clustering: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform spectral clustering of the graph nodes and return the indices of the
        centroids of the clusters and the labels of the nodes.

        Parameters:
        - n (int): Number of clusters.
        - new_clustering (bool): If True, the clusters are always recomputed.

        Returns:
        - tuple: Indices of the centroids of the clusters, labels of the data points.
        """
        if n <= 0 or n >= self.n_nodes:
            raise ValueError(
                f"Invalid number of clusters: {n}. Must be positive and less than {self.n_nodes}."
            )

        if self._should_return_previous_clusters(n, new_clustering):
            return self.inds_centroids, self.labels

        self.n_clusters = n
        _, eigvecs = self.laplacian_eig()
        self.inds_centroids, self.labels = spectral_clustering(eigvecs, n)
        self._is_graph_clustered = True

        return self.inds_centroids, self.labels

    def _should_return_previous_clusters(
        self, n: int, new_clustering: bool
    ) -> bool:
        """
        Determine if the previous clusters should be returned or if a
        new clustering needs to be performed.
        """
        if not self._is_graph_clustered:
            return False

        if n == self.n_clusters and not new_clustering:
            logger.info(
                "Spectral clustering was already performed with the"
                "same number of clusters. "
                "Returning previous clusters."
            )
            return True

        if n != self.n_clusters:
            logger.warning(
                "Clusters have already been computed with a different 'n'recomputing..."
            )

        return False

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.nodes[idx, ...]

    def __len__(self) -> int:
        return self.n_nodes


class MultiFidelityModel:
    """
    MultiFidelityModel class performs multi-fidelity modeling
    using specMF method.

    Parameters:
    - sigma (float): Noise level of high-fidelity data. Default is 1e-2.
    - beta (int): Regularization exponent. Controls the smoothness of the
        correction applied to the low-fidelity data to obtain the
        multi-fidelity data. Higher values lead to smoother solutions.
        Default is 2.
    - kappa (float): Regularization strength. Controls the weight of the prior
        with respect to the likelihood in the Bayesian update used to compute
        the multi-fidelity data. Default is 1e-3.
    - method (str): SpecMF variation to computing multi-fidelity data. Can
        be 'full' or 'trunc'. Default is 'full'.
    - spectrum_cutoff (bool): Number of eigenvectors used if method is 'trunc'.

    Methods:
    - transform(): Compute multi-fidelity data from low-fidelity graph.
    - fit_transform(): Fit omega and return the multi-fidelity data.
    - summary(): Print the model configuration.
    """

    # Numerical constants
    REG_EPS = 1e-8
    DEFAULT_SPECTRAL_GAP_EIGVALS = (
        50  # Number of eigenvalues for spectral gap computation
    )

    # Configuration constants
    _contained_params = [  # noqa: RUF012
        "sigma",
        "beta",
        "kappa",
        "method",
        "spectrum_cutoff",
        "tau",
        "omega",
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

        self._is_fit = False
        self.regularized_laplacian = None

        self._check_config()

    def transform(
        self,
        g_LF: Graph,
        x_HF: np.ndarray,
        inds_train: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute multi-fidelity data from low-fidelity graph and
        high-fidelity data.

        Parameters
        ----------
        g_LF : Graph
            The low-fidelity graph with nodes of shape (n_samples_LF, n_features).
        x_HF : np.ndarray
            The high-fidelity data of shape (n_samples_HF, n_features).
        inds_train : List[int]
            Indices providing a one-to-one mapping between high-fidelity samples in x_HF
            and low-fidelity nodes in g_LF.nodes. That is, x_HF[i] corresponds to
            g_LF.nodes[inds_train[i]].

        Returns
        -------
        x_MF : np.ndarray
            Multi-fidelity data of shape (n_samples_LF, n_features).
        C_phi : np.ndarray
            Covariance matrix of the multi-fidelity estimates.
        dPhi : np.ndarray
            Standard deviation of the multi-fidelity estimates.
        """
        # Validate inputs
        if not isinstance(g_LF, Graph):
            raise TypeError(f"Expected Graph, got {type(g_LF)}")
        validate_array_shape(x_HF, 2, "x_HF")
        validate_array_compatibility(
            g_LF.nodes, x_HF, axis=1, name1="g_LF.nodes", name2="x_HF"
        )

        if len(inds_train) != x_HF.shape[0]:
            raise ValueError(
                f"Number of high-fidelity data points does not match the number of indices. "
                f"Got {x_HF.shape[0]} high-fidelity points and {len(inds_train)} indices."
            )

        if self.tau is None:
            eigvals, _ = g_LF.laplacian_eig()
            self.tau = self._compute_spectral_gap(eigvals)

        if self.omega is None:
            self.omega = self.kappa / (self.tau**self.beta)

        if self.method == "full":
            return self._compute_specmf_data(g_LF, x_HF, inds_train)
        elif self.method == "trunc":
            return self._compute_specmf_data_trunc(g_LF, x_HF, inds_train)

    def fit_transform(
        self,
        g_LF: Graph,
        x_HF: np.ndarray,
        inds_train: list[int] | None = None,
        r: float = 3.0,
        maxiter: int = 10,
        step_size: float = 1.0,
        step_decay_rate: float = 0.999,
        ftol: float = 1e-6,
        gtol: float = 1e-8,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], list[float]]:
        """
        Optimize hyperparameter kappa and compute multi-fidelity data.

        This method finds the optimal kappa value such that the ratio between
        mean multi-fidelity uncertainty and high-fidelity noise matches the
        specified target ratio r.
        The optimization is performed in log-space using gradient descent.

        Parameters
        ----------
        g_LF : Graph
            The low-fidelity graph with nodes of shape (n_samples_LF, n_features).
        x_HF : np.ndarray
            The high-fidelity data of shape (n_samples_HF, n_features).
        inds_train : Optional[List[int]]
            Indices providing a one-to-one mapping between high-fidelity samples in x_HF
            and low-fidelity nodes in g_LF.nodes.
        r : float, default=3.0
            Target ratio between mean multi-fidelity uncertainty and high-fidelity noise level.
        maxiter : int, default=10
            Maximum number of optimization iterations.
        step_size : float, default=1.0
            Initial step size for gradient descent.
        step_decay_rate : float, default=0.999
            Multiplicative decay rate for the step size at each iteration.
        ftol : float, default=1e-6
            Loss function tolerance. Optimization stops when loss < ftol.
        gtol : float, default=1e-8
            Gradient tolerance. Optimization stops when |gradient| < gtol.
        verbose : bool, default=False
            If True, log optimization progress at each iteration.

        Returns
        -------
        x_MF : np.ndarray
            Multi-fidelity data of shape (n_samples_LF, n_features).
        C : np.ndarray
            Covariance matrix of the multi-fidelity estimates.
        dPhi : np.ndarray
            Standard deviation of the multi-fidelity estimates.
        loss_history : List[float]
            Loss value at each iteration.
        kappa_history : List[float]
            Kappa value at each iteration.

        Notes
        -----
        The loss function is: (mean(dPhi) - r * sigma)^2
        Optimization is performed in log(kappa) space for better numerical stability.
        """

        if self._is_fit:
            logger.warning(
                "Model has already been fitted. "
                "Fitting will continue from the last state. "
                "To start a new fitting, reset the 'kappa' parameter, or create a new instance of the model."
            )

        if self.kappa is None:
            raise ValueError(
                "Initial value for 'kappa' must be provided for fitting."
            )

        if self.omega is not None:
            raise ValueError(
                "Parameter 'omega' is not None; fitting would override the current value of 'omega'. "
                "Reset 'omega' to None to use the fitting method."
            )

        if self.tau is None:
            eigvals, _ = g_LF.laplacian_eig()
            self.tau = self._compute_spectral_gap(eigvals)

        if self.regularized_laplacian is None:
            self.regularized_laplacian = self._compute_regularized_laplacian(
                g_LF.graph_laplacian
            )

        # Initialize log(kappa)
        log_kappa = np.log(self.kappa)

        loss_history = []
        kappa_history = []

        for it in range(maxiter):
            # Update kappa and reset omega
            self.kappa = np.exp(log_kappa)
            self.omega = None

            # Compute the convariance matrix
            x_MF, C, dPhi = self.transform(g_LF, x_HF, inds_train)

            # Compute loss and gradient
            loss = self._compute_loss(dPhi, r)
            grad = self._compute_gradient(dPhi, C, r)

            # Store loss and kappa values
            loss_history.append(loss)
            kappa_history.append(self.kappa)

            if verbose:
                logger.info(
                    f"Iteration: {it + 1}, Loss: {loss}, Gradient: {grad}, Kappa: {self.kappa}"
                )

            if it > 0 and (loss < ftol or abs(grad) < gtol):
                break

            # Update log_kappa and step size
            log_kappa -= step_size * grad
            step_size *= step_decay_rate

        if verbose:
            logger.info(f"---- Completed after {it + 1} iterations.")
            logger.info(f"Final Loss: {loss}")

        self._is_fit = True
        return x_MF, C, dPhi, loss_history, kappa_history

    def summary(self, params_to_print: list[str] | None = None) -> None:
        """
        Print the model configuration.

        Parameters:
        - params_to_print (list): List of parameters to print. Default is all parameters.
        """
        if params_to_print is None:
            params_to_print = self._contained_params

        max_key_length = max(len(key) for key in params_to_print)
        divider = "=" * 3 * max_key_length

        print(divider)
        print("Model Configuration:")
        print(divider)

        for key, value in self.__dict__.items():
            if key in params_to_print:
                print(f"{key.ljust(max_key_length + 4)}: {value}")

        print(divider)

    def _compute_correction_residual(
        self, x_LF: np.ndarray, x_HF: np.ndarray, inds_train: list[int]
    ) -> np.ndarray:
        """
        Compute the correction residual (difference between HF and LF at training points).

        Parameters
        ----------
        x_LF : np.ndarray
            Low-fidelity data.
        x_HF : np.ndarray
            High-fidelity data.
        inds_train : List[int]
            Training indices.

        Returns
        -------
        Phi_hat : np.ndarray
            Correction residual.
        """
        return x_HF - x_LF[inds_train, :]

    def _create_selection_matrix(
        self, n_total: int, n_selected: int, inds_selected: list[int]
    ) -> np.ndarray:
        """
        Create a selection matrix that picks out selected indices.

        Parameters
        ----------
        n_total : int
            Total number of samples.
        n_selected : int
            Number of selected samples.
        inds_selected : List[int]
            Indices of selected samples.

        Returns
        -------
        P : np.ndarray
            Selection matrix of shape (n_selected, n_total).
        """
        P = np.zeros((n_selected, n_total))
        P[np.arange(n_selected), inds_selected] = 1
        return P

    def _compute_specmf_data(
        self, g_LF: Graph, x_HF: np.ndarray, inds_train: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute multi-fidelity data using standard specMF method.

        Parameters
        ----------
        g_LF : Graph
            The low-fidelity graph with nodes of shape (n_samples_LF, n_features).
        x_HF : np.ndarray
            The high-fidelity data of shape (n_samples_HF, n_features).
        inds_train : List[int]
            Indices providing one-to-one mapping between HF and LF data.

        Returns
        -------
        x_MF : np.ndarray
            Multi-fidelity data of shape (n_samples_LF, n_features).
        C_phi : np.ndarray
            Covariance matrix.
        dPhi : np.ndarray
            Standard deviation of estimates.
        """
        x_LF = g_LF.nodes
        n_LF, n_HF = x_LF.shape[0], len(inds_train)

        # Compute correction residual
        Phi_hat = self._compute_correction_residual(x_LF, x_HF, inds_train)

        # Create selection matrix
        P_N = self._create_selection_matrix(n_LF, n_HF, inds_train)

        # Compute or retrieve regularized Laplacian
        if self.regularized_laplacian is None:
            self.regularized_laplacian = self._compute_regularized_laplacian(
                g_LF.graph_laplacian
            )

        # Construct and solve the system
        B = (1 / self.sigma**2) * (
            P_N.T @ P_N
        ) + self.omega * self.regularized_laplacian
        C_phi = solve(B, np.eye(n_LF))
        Phi_mean = (1 / self.sigma**2) * C_phi @ P_N.T @ Phi_hat

        # Compute multi-fidelity estimates
        x_MF = x_LF + Phi_mean
        dPhi = np.sqrt(np.diag(C_phi) + self.REG_EPS)

        return x_MF, C_phi, dPhi

    def _compute_specmf_data_trunc(
        self, g_LF: Graph, x_HF: np.ndarray, inds_train: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute multi-fidelity data using truncated specMF method.

        The truncated method expresses multi-fidelity corrections as a linear combination
        of low-lying eigenvectors of the graph Laplacian, reducing computational cost.

        Parameters
        ----------
        g_LF : Graph
            The low-fidelity graph with nodes of shape (n_samples_LF, n_features).
        x_HF : np.ndarray
            The high-fidelity data of shape (n_samples_HF, n_features).
        inds_train : List[int]
            Indices providing one-to-one mapping between HF and LF data.

        Returns
        -------
        x_MF : np.ndarray
            Multi-fidelity data of shape (n_samples_LF, n_features).
        C_phi : np.ndarray
            Covariance matrix.
        dPhi : np.ndarray
            Standard deviation of estimates.

        Notes
        -----
        The number of eigenvectors used is controlled by self.spectrum_cutoff.
        """
        x_LF = g_LF.nodes
        eigvals, eigvecs = g_LF.laplacian_eig()

        # Truncate to low-lying eigenvectors
        Psi = eigvecs[:, : self.spectrum_cutoff]
        Psi_N = Psi[inds_train, :]

        # Compute correction residual
        Phi_hat = self._compute_correction_residual(x_LF, x_HF, inds_train)

        # Regularize truncated eigenvalues
        eigvals_reg = (
            np.abs(eigvals[: self.spectrum_cutoff]) + self.tau
        ) ** self.beta

        # Construct and solve reduced system
        B = (1 / self.sigma**2) * (Psi_N.T @ Psi_N) + self.omega * np.diag(
            eigvals_reg
        )
        C_a = solve(B, np.eye(self.spectrum_cutoff))
        A = (1 / self.sigma**2) * C_a @ Psi_N.T @ Phi_hat

        # Project back to full space
        Phi_mean = Psi @ A
        x_MF = x_LF + Phi_mean
        C_phi = Psi @ C_a @ Psi.T
        dPhi = np.sqrt(np.diag(C_phi) + self.REG_EPS)

        return x_MF, C_phi, dPhi

    def _compute_spectral_gap(self, eigvals: np.ndarray) -> float:
        """Compute spectral gap for tau estimation.

        Parameters:
        - eigvals (np.ndarray): Eigenvalues of the graph Laplacian.

        Returns:
        - float: The eigenvalue corresponding to the highest curvature of the spectrum in log-scale.
        """
        eigvals_ = np.abs(eigvals[: self.DEFAULT_SPECTRAL_GAP_EIGVALS])
        log_eigvals = np.log10(eigvals_)
        log_curvature = (
            log_eigvals[:-2] + log_eigvals[2:] - 2 * log_eigvals[1:-1]
        )
        return eigvals[np.argmin(log_curvature) + 1]

    def _compute_regularized_laplacian(self, L: np.ndarray) -> np.ndarray:
        """Compute the regularized graph Laplacian.

        Parameters:
        - L (np.ndarray): Graph Laplacian matrix.

        Returns:
        - np.ndarray: Regularized Laplacian matrix.
        """
        return np.linalg.matrix_power(
            L + self.tau * np.eye(L.shape[0]), self.beta
        )

    def _compute_loss(self, dPhi: np.ndarray, r: float) -> float:
        """
        Compute loss function. This is equal to squared difference between the mean multi-fidelity
        uncertainty and high-fidelity noise.

        Parameters:
        - dPhi (np.ndarray): Multi-fidelity estimates uncertainty.
        - r (float): The ratio of the mean multi-fidelity uncertainty to the high-fidelity noise level.

        Returns:
        - float: The loss value.
        """
        return (np.mean(dPhi) - r * self.sigma) ** 2

    def _compute_gradient(
        self, dPhi: np.ndarray, C: np.ndarray, r: float
    ) -> float:
        """
        Compute the gradient of the loss with respect to log(kappa).

        Parameters:
        - dPhi (np.ndarray): Multi-fidelity estimates uncertainty.
        - C (np.ndarray): Multi-fidelity estimates covariance matrix.
        - r (float): The ratio of the mean multi-fidelity uncertainty to the high-fidelity noise level.

        Returns:
        - float: The gradient value.
        """
        dloss_dC = (
            (1 / dPhi.size) * (np.mean(dPhi) - r * self.sigma) * (1 / dPhi)
        )
        dC_dkappa = -(1 / self.tau**self.beta) * np.einsum(
            "ij,ij->j", C, self.regularized_laplacian @ C
        )
        dkappa_dlogkappa = self.kappa
        return np.sum(dloss_dC * dC_dkappa) * dkappa_dlogkappa

    def _check_config(self) -> None:
        """Validate model configuration parameters."""
        validate_method_choice(self.method, ["full", "trunc"], "method")

        if self.method == "trunc" and self.spectrum_cutoff is None:
            raise ValueError(
                "With method 'trunc', parameter 'spectrum_cutoff' must be provided."
            )
        if self.method == "full" and self.spectrum_cutoff is not None:
            logger.warning(
                "When method is 'full' the parameter 'spectrum_cutoff' is ignored."
            )

        validate_positive_scalar(self.sigma, "sigma")
        validate_positive_scalar(self.beta, "beta")

        if self.spectrum_cutoff is not None:
            validate_positive_scalar(self.spectrum_cutoff, "spectrum_cutoff")

        if self.omega is not None:
            validate_positive_scalar(self.omega, "omega")
            if self.kappa is not None:
                logger.warning(
                    "Parameter 'kappa' is ignored when 'omega' is provided."
                )

        if self.omega is None:
            validate_positive_scalar(self.kappa, "kappa")

        if self.tau is not None:
            validate_positive_scalar(self.tau, "tau")
