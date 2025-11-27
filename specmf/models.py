"""Multi-fidelity modeling using spectral methods."""

import logging
from functools import cached_property

import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize

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
    """Graph representation with spectral properties and clustering.

    This class extends GraphCore to store graph representations and
    quantities like the adjacency matrix, graph Laplacian, and
    eigenvectors and eigenvalues of the graph Laplacian.

    Methods
    -------
    adjacency : cached_property
        Compute and return the adjacency matrix.
    graph_laplacian : cached_property
        Compute and return the graph Laplacian.
    laplacian_eig()
        Return eigenvalues and eigenvectors of graph Laplacian.
    cluster(n, new_clustering)
        Perform spectral clustering and return cluster info.
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
        """Return eigenvalues and eigenvectors of graph Laplacian.

        Returns
        -------
        tuple of np.ndarray
            Eigenvalues and eigenvectors of the graph Laplacian.
        """
        if self.eigvals is None or self.eigvecs is None:
            self.eigvals, self.eigvecs = ordered_eig(self.graph_laplacian)
        return self.eigvals, self.eigvecs

    def cluster(
        self, n: int, new_clustering: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform spectral clustering of graph nodes.

        Parameters
        ----------
        n : int
            Number of clusters.
        new_clustering : bool, optional
            If True, clusters are always recomputed. Default is False.

        Returns
        -------
        tuple of np.ndarray
            Indices of cluster centroids and labels of data points.
        """
        if n <= 0 or n >= self.n_nodes:
            raise ValueError(
                f"Invalid number of clusters: {n}. "
                f"Must be positive and less than {self.n_nodes}."
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
        """Determine if previous clusters should be returned.

        Returns True if clustering already done with same parameters.
        """
        if not self._is_graph_clustered:
            return False

        if n == self.n_clusters and not new_clustering:
            logger.info(
                "Spectral clustering was already performed with the "
                "same number of clusters. Returning previous clusters."
            )
            return True

        if n != self.n_clusters:
            logger.warning(
                "Clusters have already been computed with a different "
                "'n' recomputing..."
            )

        return False

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.nodes[idx, ...]

    def __len__(self) -> int:
        return self.n_nodes


class MultiFidelityModel:
    """Multi-fidelity modeling using spectral methods (specMF).

    Parameters
    ----------
    sigma : float, optional
        Noise level of high-fidelity data. Default is 1e-2.
    beta : int, optional
        Regularization exponent controlling smoothness of correction.
        Higher values lead to smoother solutions. Default is 2.
    kappa : float, optional
        Regularization strength controlling weight of prior vs
        likelihood. Default is 1e-3.
    omega : float, optional
        Alternative to kappa for direct regularization control.
    method : str, optional
        SpecMF variation: 'full' or 'trunc'. Default is 'full'.
    spectrum_cutoff : int, optional
        Number of eigenvectors used if method='trunc'.
    tau : float, optional
        Spectral gap parameter for regularization.

    Methods
    -------
    transform(g_LF, x_HF, inds_train)
        Compute multi-fidelity data from low-fidelity graph.
    fit_transform(g_LF, x_HF, inds_train, ...)
        Optimize hyperparameters and compute multi-fidelity data.
    summary(params_to_print)
        Print the model configuration.
    """

    # Numerical constants
    REG_EPS = 1e-8
    # Number of eigenvalues for spectral gap computation
    DEFAULT_SPECTRAL_GAP_EIGVALS = 50

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
        omega: float | None = None,
        method: str = "full",
        spectrum_cutoff: int | None = None,
        tau: float | None = None,
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
            The low-fidelity graph with nodes of shape
            (n_samples_LF, n_features).
        x_HF : np.ndarray
            The high-fidelity data of shape (n_samples_HF, n_features).
        inds_train : List[int]
            Indices providing a one-to-one mapping between high-fidelity
            samples in x_HF and low-fidelity nodes in g_LF.nodes.
            That is, x_HF[i] corresponds to g_LF.nodes[inds_train[i]].

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
                f"Number of high-fidelity data points does not match "
                f"the number of indices. Got {x_HF.shape[0]} "
                f"high-fidelity points and {len(inds_train)} indices."
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
        maxiter: int = 100,
        ftol: float = 1e-6,
        gtol: float = 1e-9,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], list[float]]:
        """
        Optimize hyperparameter kappa and compute multi-fidelity data.

        This method finds the optimal kappa value such that the ratio between
        mean multi-fidelity uncertainty and high-fidelity noise matches the
        specified target ratio r.
        The optimization is performed in log-space using L-BFGS-B.

        Parameters
        ----------
        g_LF : Graph
            The low-fidelity graph with nodes of shape
            (n_samples_LF, n_features).
        x_HF : np.ndarray
            The high-fidelity data of shape (n_samples_HF, n_features).
        inds_train : Optional[List[int]]
            Indices providing a one-to-one mapping between high-fidelity
            samples in x_HF and low-fidelity nodes in g_LF.nodes.
        r : float, default=3.0
            Target ratio between mean multi-fidelity uncertainty and
            high-fidelity noise level.
        maxiter : int, default=10
            Maximum number of optimization iterations.
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
        Optimization is performed in log(kappa) space for better numerical
        stability.
        """

        if self._is_fit:
            logger.warning(
                "Model has already been fitted. "
                "Fitting will continue from the last state. "
                "To start a new fitting, reset the 'kappa' parameter, "
                "or create a new instance of the model."
            )

        if self.kappa is None:
            raise ValueError(
                "Initial value for 'kappa' must be provided for fitting."
            )

        if self.omega is not None:
            raise ValueError(
                "Parameter 'omega' is not None; fitting would "
                "override the current value of 'omega'. "
                "Reset 'omega' to None to use the fitting method."
            )

        if self.tau is None:
            eigvals, _ = g_LF.laplacian_eig()
            self.tau = self._compute_spectral_gap(eigvals)

        if self.regularized_laplacian is None:
            self.regularized_laplacian = self._compute_regularized_laplacian(
                g_LF.graph_laplacian
            )

        # histories to return
        loss_history: list[float] = []
        kappa_history: list[float] = []

        # Objective function in log(kappa) space
        def objective(log_kappa: np.ndarray) -> tuple[float, np.ndarray]:
            """Return loss and gradient wrt log(kappa) for SciPy."""
            # log_kappa comes as array([value])
            log_kappa_scalar = float(log_kappa)

            # update model params
            self.kappa = np.exp(log_kappa_scalar)
            self.omega = None  # force recomputation from kappa, tau, beta

            # forward pass
            _, C, dPhi = self.transform(g_LF, x_HF, inds_train)

            # loss + gradient - already wrt log(kappa)
            loss = self._compute_loss(dPhi, r)
            grad = self._compute_gradient(dPhi, C, r)

            # update histories
            loss_history.append(loss)
            kappa_history.append(self.kappa)

            if verbose:
                logger.info(
                    "log(kappa)=%.3e, kappa=%.3e, loss=%.3e, grad=%.3e",
                    log_kappa_scalar,
                    self.kappa,
                    loss,
                    grad,
                )

            return loss, np.array([grad], dtype=float)

        # initial point in log-space
        x0 = np.array([np.log(self.kappa)], dtype=float)

        # bounds for log-kappa
        log_kappa_bounds = [(np.log(1e-8), np.log(1e8))]

        res = minimize(
            fun=lambda z: objective(z)[0],
            x0=x0,
            jac=lambda z: objective(z)[1],
            method="L-BFGS-B",
            bounds=log_kappa_bounds,
            options={
                "maxiter": maxiter,
                "ftol": ftol,
                "gtol": gtol,
            },
        )

        # final kappa from optimizer
        log_kappa_opt = float(res.x[0])
        self.kappa = np.exp(log_kappa_opt)
        self.omega = None

        if verbose:
            logger.info("Optimization success: %s", res.success)
            logger.info(
                "Final log(kappa)=%.3e, kappa=%.3e", log_kappa_opt, self.kappa
            )
            logger.info("Final loss=%.3e", res.fun)

        # final forward pass with optimal kappa
        x_MF, C, dPhi = self.transform(g_LF, x_HF, inds_train)
        self._is_fit = True

        return x_MF, C, dPhi, loss_history, kappa_history

    def summary(self, params_to_print: list[str] | None = None) -> None:
        """Print the model configuration.

        Parameters
        ----------
        params_to_print : list of str, optional
            List of parameters to print. Default is all parameters.
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
        self,
        x_LF: np.ndarray,
        x_HF: np.ndarray,
        inds_train: list[int],
    ) -> np.ndarray:
        """Compute correction residual (HF - LF at training points).

        Parameters
        ----------
        x_LF : np.ndarray
            Low-fidelity data.
        x_HF : np.ndarray
            High-fidelity data.
        inds_train : list of int
            Training indices.

        Returns
        -------
        np.ndarray
            Correction residual.
        """
        return x_HF - x_LF[inds_train, :]

    def _create_selection_matrix(
        self,
        n_total: int,
        n_selected: int,
        inds_selected: list[int],
    ) -> np.ndarray:
        """Create a selection matrix that picks selected indices.

        Parameters
        ----------
        n_total : int
            Total number of samples.
        n_selected : int
            Number of selected samples.
        inds_selected : list of int
            Indices of selected samples.

        Returns
        -------
        np.ndarray
            Selection matrix of shape (n_selected, n_total).
        """
        P = np.zeros((n_selected, n_total))
        P[np.arange(n_selected), inds_selected] = 1
        return P

    def _compute_specmf_data(
        self,
        g_LF: Graph,
        x_HF: np.ndarray,
        inds_train: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute multi-fidelity data using standard specMF method.

        Parameters
        ----------
        g_LF : Graph
            Low-fidelity graph with nodes (n_samples_LF, n_features).
        x_HF : np.ndarray
            High-fidelity data of shape (n_samples_HF, n_features).
        inds_train : list of int
            Indices providing one-to-one mapping HF to LF.

        Returns
        -------
        x_MF : np.ndarray
            Multi-fidelity data (n_samples_LF, n_features).
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
            L_reg = self._compute_regularized_laplacian(g_LF.graph_laplacian)
            self.regularized_laplacian = L_reg

        # Construct and solve the system
        B = (1 / self.sigma**2) * (P_N.T @ P_N)
        B = B + self.omega * self.regularized_laplacian
        C_phi = solve(B, np.eye(n_LF))
        Phi_mean = (1 / self.sigma**2) * C_phi @ P_N.T @ Phi_hat

        # Compute multi-fidelity estimates
        x_MF = x_LF + Phi_mean
        dPhi = np.sqrt(np.diag(C_phi) + self.REG_EPS)

        return x_MF, C_phi, dPhi

    def _compute_specmf_data_trunc(
        self,
        g_LF: Graph,
        x_HF: np.ndarray,
        inds_train: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute multi-fidelity data using truncated specMF method.

        The truncated method expresses multi-fidelity corrections as a
        linear combination of low-lying eigenvectors of the graph
        Laplacian, reducing computational cost.

        Parameters
        ----------
        g_LF : Graph
            Low-fidelity graph with nodes (n_samples_LF, n_features).
        x_HF : np.ndarray
            High-fidelity data of shape (n_samples_HF, n_features).
        inds_train : list of int
            Indices providing one-to-one mapping HF to LF.

        Returns
        -------
        x_MF : np.ndarray
            Multi-fidelity data (n_samples_LF, n_features).
        C_phi : np.ndarray
            Covariance matrix.
        dPhi : np.ndarray
            Standard deviation of estimates.

        Notes
        -----
        Number of eigenvectors used controlled by spectrum_cutoff.
        """
        x_LF = g_LF.nodes
        eigvals, eigvecs = g_LF.laplacian_eig()

        # Truncate to low-lying eigenvectors
        Psi = eigvecs[:, : self.spectrum_cutoff]
        Psi_N = Psi[inds_train, :]

        # Compute correction residual
        Phi_hat = self._compute_correction_residual(x_LF, x_HF, inds_train)

        # Regularize truncated eigenvalues
        eigvals_trunc = np.abs(eigvals[: self.spectrum_cutoff])
        eigvals_reg = (eigvals_trunc + self.tau) ** self.beta

        # Construct and solve reduced system
        B = (1 / self.sigma**2) * (Psi_N.T @ Psi_N)
        B = B + self.omega * np.diag(eigvals_reg)
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

        Parameters
        ----------
        eigvals : np.ndarray
            Eigenvalues of the graph Laplacian.

        Returns
        -------
        float
            Eigenvalue at highest curvature of spectrum in log-scale.
        """
        n_eigs = self.DEFAULT_SPECTRAL_GAP_EIGVALS
        eigvals_ = np.abs(eigvals[:n_eigs])
        log_eigvals = np.log10(eigvals_)
        log_curvature = (
            log_eigvals[:-2] + log_eigvals[2:] - 2 * log_eigvals[1:-1]
        )
        return eigvals[np.argmin(log_curvature) + 1]

    def _compute_regularized_laplacian(self, L: np.ndarray) -> np.ndarray:
        """Compute the regularized graph Laplacian.

        Parameters
        ----------
        L : np.ndarray
            Graph Laplacian matrix.

        Returns
        -------
        np.ndarray
            Regularized Laplacian matrix.
        """
        L_reg = L + self.tau * np.eye(L.shape[0])
        return np.linalg.matrix_power(L_reg, self.beta)

    def _compute_loss(self, dPhi: np.ndarray, r: float) -> float:
        """Compute loss function for hyperparameter optimization.

        Loss is squared difference between mean multi-fidelity
        uncertainty and high-fidelity noise.

        Parameters
        ----------
        dPhi : np.ndarray
            Multi-fidelity estimates uncertainty.
        r : float
            Target ratio of mean MF uncertainty to HF noise level.

        Returns
        -------
        float
            The loss value.
        """
        return (np.mean(dPhi) - r * self.sigma) ** 2

    def _compute_gradient(
        self, dPhi: np.ndarray, C: np.ndarray, r: float
    ) -> float:
        """Compute gradient of loss with respect to log(kappa).

        Parameters
        ----------
        dPhi : np.ndarray
            Multi-fidelity estimates uncertainty.
        C : np.ndarray
            Multi-fidelity estimates covariance matrix.
        r : float
            Target ratio of mean MF uncertainty to HF noise level.

        Returns
        -------
        float
            The gradient value.
        """
        dloss_dC = (
            (1 / dPhi.size) * (np.mean(dPhi) - r * self.sigma) * (1 / dPhi)
        )
        dC_dkappa = -(1 / self.tau**self.beta) * np.einsum(
            "ij,ij->j", C, self.regularized_laplacian @ C
        )
        dkappa_dlogkappa = self.kappa
        gradient = np.sum(dloss_dC * dC_dkappa) * dkappa_dlogkappa
        return gradient

    def _check_config(self) -> None:
        """Validate model configuration parameters."""
        validate_method_choice(self.method, ["full", "trunc"], "method")

        if self.method == "trunc" and self.spectrum_cutoff is None:
            raise ValueError(
                "With method 'trunc', parameter 'spectrum_cutoff' "
                "must be provided."
            )
        if self.method == "full" and self.spectrum_cutoff is not None:
            logger.warning(
                "When method is 'full' the parameter "
                "'spectrum_cutoff' is ignored."
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
