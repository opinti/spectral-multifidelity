import logging
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


# Setup logging
logger = logging.getLogger(__name__)


# Update matplotlib settings
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
})


# General plotting functions
def plot_distributions(
    e_LF: np.ndarray,
    e_MF: np.ndarray,
    bins_LF: int = 50,
    bins_MF: int = 50,
    mask: np.ndarray | None = None,
    return_axs: bool = False,
) -> np.ndarray | None:
    if mask is None:
        mask = np.ones_like(e_LF, dtype=bool)

    fig, axs = plt.subplots(
        2, 1, figsize=(10, 5), gridspec_kw={"height_ratios": [3, 1]}
    )

    f, _, _ = axs[0].hist(
        e_MF[mask], bins=bins_MF, alpha=0.75, zorder=10, label="Multi-Fidelity"
    )
    axs[0].hist(e_LF[mask], bins=bins_LF, alpha=0.75, label="Low-Fidelity")
    axs[0].grid(True, linewidth=0.5)
    f_max = 100 * (f.max() // 100)
    axs[0].set_yticks([f_max // 4, f_max // 2, 3 * f_max // 4, f_max])
    plt.setp(
        axs[0].get_xticklabels(),
        visible=False,
    )
    plt.setp(
        axs[0].get_yticklabels(),
        fontsize=16,
    )
    axs[0].legend(fontsize=20)

    axs[1].plot(e_MF[mask], [0.25] * len(e_LF[mask]), "|")
    axs[1].plot(e_LF[mask], [0.75] * len(e_LF[mask]), "|")
    axs[1].set_yticks([0, 1])
    plt.setp(
        axs[1].get_xticklabels(),
        fontsize=16,
    )
    plt.setp(axs[1].get_yticklabels(), visible=False)
    axs[1].set_xlabel(r"Error [\%]", fontsize=26, labelpad=15)
    plt.tight_layout()

    return axs if return_axs else None


def plot_spectrum(eigvals: np.ndarray, n: int) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(
        np.arange(1, n + 1), np.abs(eigvals[:n]), "o-", c="k", markersize=5
    )
    ax.set_yscale("log")
    ax.set_xticks(range(1, n + 1, 5))
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlabel("$m$", fontsize=20)
    ax.set_ylabel(r"$\lambda_m$", fontsize=20, rotation=0, labelpad=20)
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()


def plot_cluster_size_hist(labels: np.ndarray) -> None:
    """Plot a histogram of cluster sizes."""
    clus_sizes = np.bincount(labels)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.sort(clus_sizes), bins=20)
    ax.set_xlabel("Cluster size", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title("Cluster size histogram", fontsize=18)
    plt.tight_layout()


def plot_loss_and_kappa(
    loss_history: list[float], kappa_history: list[float]
) -> None:
    """
    Plot loss history and kappa history on the same figure with two y-axes.
    """
    assert len(loss_history) == len(kappa_history), (
        "Loss and kappa histories must have the same length."
    )

    iterations = np.arange(len(loss_history))

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot loss history
    ax1.plot(iterations, loss_history, "b-", label="Loss")
    ax1.set_yscale("log")
    ax1.set_xlabel("Iteration", fontsize=14)
    ax1.set_ylabel("Loss", color="b", fontsize=14)
    ax1.tick_params(axis="y", labelcolor="b")

    # Plot kappa history
    ax2 = ax1.twinx()
    ax2.plot(iterations, kappa_history, "r-", label=r"$\kappa$")
    ax2.set_yscale("log")
    ax2.set_ylabel(
        r"$\kappa$", color="r", rotation=0, labelpad=20, fontsize=16
    )
    ax2.tick_params(axis="y", labelcolor="r")

    ax1.grid(True)
    plt.title("Loss and Kappa History")
    plt.show()


# Data plotting functions
def plot_data(
    X_LF: np.ndarray, X_HF: np.ndarray, dataset_name: str, **kwargs
) -> None:
    """Plot low-fidelity and high-fidelity data based on the dataset name."""
    plotters: dict[str, Callable] = {
        "elasticity-displacement": _plot_lf_hf_fields_samples,
        "darcy-flow": _plot_lf_hf_fields_samples,
        "elasticity-traction": _plot_lf_hf_qoi_samples,
        "beam": _plot_lf_hf_curves_samples,
        "cavity-flow": _plot_lf_hf_curves_samples,
    }

    plot_func = plotters.get(dataset_name)
    if plot_func is None:
        raise ValueError(
            f"Invalid dataset name: {dataset_name}. Expected one of {list(plotters.keys())}."
        )

    plot_func(X_LF, X_HF, **kwargs)
    plt.show()


def _plot_lf_hf_fields_samples(
    X_LF: np.ndarray,
    X_HF: np.ndarray,
    n_samples: int = 5,
    levels: int = 10,
    **kwargs,
) -> None:
    """Plot field samples for low-fidelity and high-fidelity datasets."""
    n_points, n_dim = X_LF.shape
    n_res = int(np.sqrt(n_dim))

    for _ in range(n_samples):
        j = np.random.randint(0, n_points)
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.subplots_adjust(wspace=0.15)

        titles = ["Low-fidelity", "High-fidelity", "Difference"]
        for ax, title in zip(axs, titles, strict=False):
            ax.set_title(title, fontsize=20)

        vmin, vmax = (
            np.min([X_LF[j, :], X_HF[j, :]]),
            np.max([X_LF[j, :], X_HF[j, :]]),
        )
        lf = axs[0].contourf(
            X_LF[j, :].reshape(n_res, n_res),
            levels=levels,
            vmin=vmin,
            vmax=vmax,
        )
        hf = axs[1].contourf(
            X_HF[j, :].reshape(n_res, n_res),
            levels=levels,
            vmin=vmin,
            vmax=vmax,
        )
        diff = axs[2].contourf(
            np.abs(
                X_HF[j, :].reshape(n_res, n_res)
                - X_LF[j, :].reshape(n_res, n_res)
            ),
            cmap="Reds",
        )

        plt.colorbar(lf, ax=axs[0])
        plt.colorbar(hf, ax=axs[1])
        plt.colorbar(diff, ax=axs[2])


def _plot_lf_hf_curves_samples(
    X_LF: np.ndarray, X_HF: np.ndarray, n_samples: int = 5
) -> None:
    """Plot curve samples for low-fidelity and high-fidelity datasets."""
    n_points = X_LF.shape[0]
    fig, axs = plt.subplots(n_samples, 1, figsize=(5, 4 * n_samples))
    fig.subplots_adjust(wspace=0.1)

    for i in range(n_samples):
        j = np.random.randint(0, n_points)
        vmin, vmax = (
            np.min([X_LF[j, :], X_HF[j, :]]) * 1.2,
            np.max([X_LF[j, :], X_HF[j, :]]) * 1.2,
        )
        x_ = np.linspace(0, 1, X_LF[j, :].shape[0])

        axs[i].plot(x_, X_LF[j, :], label="Low-Fidelity")
        axs[i].plot(x_, X_HF[j, :], label="High-Fidelity")
        axs[i].set_ylim([vmin, vmax])
        axs[i].legend()
        axs[i].grid(True)


def _plot_lf_hf_qoi_samples(
    X_LF: np.ndarray, X_HF: np.ndarray, **kwargs
) -> None:
    """Plot low-fidelity and high-fidelity quantities of interest samples."""
    # Plot clustered dataset and centorids
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    fig.subplots_adjust(wspace=0.3)
    n_planes = X_LF.shape[1] - 1

    for i in range(n_planes):
        y_max, y_min = (
            1.2 * np.max((X_LF[:, i + 1], X_HF[:, i + 1])),
            1.2 * np.min((X_LF[:, i + 1], X_HF[:, i + 1])),
        )
        x_max, x_min = (
            1.2 * np.max((X_LF[:, i], X_HF[:, i])),
            1.2 * np.min((X_LF[:, i], X_HF[:, i])),
        )

        axs[0, i].scatter(X_LF[:, i], X_LF[:, i + 1], s=3)
        axs[0, i].set_xlabel(f"$u_{i + 1}$", fontsize=24)
        axs[0, i].set_ylabel(
            f"$u_{i + 2}$", fontsize=24, rotation=0, labelpad=15
        )
        axs[0, i].tick_params(axis="both", labelsize=16)
        axs[0, i].set_xlim((x_min, x_max))
        axs[0, i].set_ylim((y_min, y_max))

        axs[1, i].scatter(X_HF[:, i], X_HF[:, i + 1], s=3, c="r")
        axs[1, i].set_xlabel(
            f"$u_{i + 1}$",
            fontsize=24,
        )
        axs[1, i].set_ylabel(
            f"$u_{i + 2}$",
            fontsize=24,
            rotation=0,
            labelpad=15,
        )
        axs[1, i].tick_params(axis="both", labelsize=16)
        axs[1, i].set_xlim((x_min, x_max))
        axs[1, i].set_ylim((y_min, y_max))


# Comparison plotting functions
def plot_mf_comparison(
    X_LF: np.ndarray,
    X_MF: np.ndarray,
    X_HF: np.ndarray,
    dataset_name: str,
    **kwargs,
) -> None:
    """Plot multi-fidelity comparisons based on the dataset name."""
    plotters = {
        "elasticity-displacement": _plot_mf_comparison_field,
        "darcy-flow": _plot_mf_comparison_field,
        "elasticity-traction": _plot_mf_comparison_qoi,
        "beam": _plot_mf_comparison_curves,
        "cavity-flow": _plot_mf_comparison_curves,
    }

    plot_func = plotters.get(dataset_name)
    if plot_func is None:
        raise ValueError(
            f"Invalid dataset name: {dataset_name}. Expected one of {list(plotters.keys())}."
        )

    plot_func(X_LF, X_MF, X_HF, **kwargs)
    plt.show()


def _plot_mf_comparison_field(
    X_LF: np.ndarray,
    X_MF: np.ndarray,
    X_HF: np.ndarray,
    samples: list[int],
    input_field: np.ndarray | None = None,  # Permeability data, optional
    titles: list[str] | None = None,  # Titles for subplots, optional
    levels: int = 10,
    cmap_diff: str = "Reds",
    cmap_inp: str = "Blues",
    **kwargs,
) -> None:
    """
    Generalized function to plot multi-fidelity field comparisons with optional permeability data.

    Parameters:
    - X_LF: Low-fidelity data (2D array).
    - X_MF: Multi-fidelity data (2D array).
    - X_HF: High-fidelity data (2D array).
    - samples: List of samples to plot. If None, 5 random samples are chosen.
    - input_field: Optional permeability data (3D array).
    - titles: Optional list of titles for subplots.
    - n_samples: Number of samples to plot (default is 4).
    - levels: Number of contour levels (default is 10).
    - cmap_diff: Colormap for error plots (default is "Reds").
    - cmap_inp: Colormap for permeability plots (default is "Blues").
    """

    if samples is None:
        samples = np.random.choice(X_LF.shape[0], 5, replace=False)

    n_samples = len(samples)
    n_points, n_dim = X_LF.shape
    n_res = int(np.sqrt(n_dim))

    # Set default titles if none provided
    if titles is None:
        if input_field is not None:
            titles = [
                "Permeability",
                "Low-Fidelity",
                "Low-Fidelity Error",
                "Multi-Fidelity",
                "Multi-Fidelity Error",
            ]
        else:
            titles = [
                "Low-Fidelity",
                "Low-Fidelity Error",
                "Multi-Fidelity",
                "Multi-Fidelity Error",
            ]

    # Determine the number of columns
    n_cols = len(titles)

    # Create the figure and axes
    fig, axs = plt.subplots(
        n_samples, n_cols, figsize=(5 * n_cols, 4.25 * n_samples)
    )
    fig.subplots_adjust(wspace=0.1)

    # Set subplot titles
    for col, title in enumerate(titles):
        axs[0, col].set_title(title, fontsize=24, pad=20)

    # Loop through the samples and plot
    for i in range(n_samples):
        j = samples[i]

        # Plot permeability if K is provided
        if input_field is not None:
            inp_field = axs[i, 0].contourf(
                input_field[:, :, j],
                levels=levels,
                cmap=cmap_inp,
            )

        vmin, vmax = np.min([X_LF[j], X_HF[j]]), np.max([X_LF[j], X_HF[j]])
        vmax_diff = np.max([
            np.abs(X_HF[j] - X_LF[j]),
            np.abs(X_MF[j] - X_LF[j]),
        ])

        # Plot low-fidelity, multi-fidelity, and error data
        axs[i, -4].contourf(
            X_LF[j].reshape(n_res, n_res), vmin=vmin, vmax=vmax, levels=levels
        )
        lf_diff = axs[i, -3].contourf(
            np.abs(X_HF[j] - X_LF[j]).reshape(n_res, n_res),
            vmin=0,
            vmax=vmax_diff,
            cmap=cmap_diff,
            levels=levels + 2,
        )
        mf = axs[i, -2].contourf(
            X_MF[j].reshape(n_res, n_res), vmin=vmin, vmax=vmax, levels=levels
        )
        axs[i, -1].contourf(
            np.abs(X_HF[j] - X_MF[j]).reshape(n_res, n_res),
            vmin=0,
            vmax=vmax_diff,
            cmap=cmap_diff,
            levels=levels + 2,
        )

        for ax in axs[i, :]:
            ax.set_xticks([])
            ax.set_yticks([])

        def add_colorbar(ax, contour_plot, labelsize=14, nbins=5):
            """Add a colorbar to a contour plot with specified parameters."""
            cbar = plt.colorbar(contour_plot, ax=ax)
            cbar.ax.tick_params(labelsize=labelsize)
            cbar.locator = MaxNLocator(nbins=nbins)
            cbar.update_ticks()
            return cbar

        add_colorbar(axs[i, -4], mf)
        add_colorbar(axs[i, -3], lf_diff)
        add_colorbar(axs[i, -2], mf)
        add_colorbar(axs[i, -1], lf_diff)
        if input_field is not None:
            add_colorbar(axs[i, 0], inp_field)

    plt.show()


def _plot_mf_comparison_qoi(
    X_LF: np.ndarray,
    X_MF: np.ndarray,
    X_HF: np.ndarray,
    inds_centroids: list[int],
    **kwargs,
) -> None:
    """Plot multi-fidelity comparison for set of quantities of interest."""
    fig, axs = plt.subplots(4, 3, figsize=(14, 20))
    fig.subplots_adjust(wspace=0.2)

    titles = [
        "Low-Fidelity",
        "Multi-Fidelity",
        "High-Fidelity",
    ]
    for ax, title in zip(axs[0], titles, strict=False):
        ax.set_title(title, fontsize=24, pad=20)

    for i in range(4):
        y_min = 1.1 * np.min([X_LF[:, i + 1], X_MF[:, i + 1], X_HF[:, i + 1]])
        y_max = 1.1 * np.max([X_LF[:, i + 1], X_MF[:, i + 1], X_HF[:, i + 1]])
        x_min = 1.1 * np.min([X_LF[:, i], X_MF[:, i], X_HF[:, i]])
        x_max = 1.1 * np.max([X_LF[:, i], X_MF[:, i], X_HF[:, i]])

        axs[i, 0].scatter(X_LF[:, i], X_LF[:, i + 1], c="orange", s=3)
        axs[i, 0].scatter(
            X_LF[inds_centroids, i],
            X_LF[inds_centroids, i + 1],
            c="blue",
            s=10,
            label="Centroids",
        )
        axs[i, 0].set_xlabel(f"$u_{i + 1}$", fontsize=22)
        axs[i, 0].set_ylabel(
            f"$u_{i + 2}$", fontsize=22, rotation=0, labelpad=15
        )
        axs[i, 0].tick_params(axis="both", labelsize=12)
        axs[i, 0].set_xlim((x_min, x_max))
        axs[i, 0].set_ylim((y_min, y_max))
        axs[i, 0].tick_params(axis="both", labelsize=18)

        axs[i, 1].scatter(X_MF[:, i], X_MF[:, i + 1], s=3)
        axs[i, 1].set_xlabel(f"$u_{i + 1}$", fontsize=22)
        axs[i, 1].tick_params(axis="both", labelsize=12)
        axs[i, 1].set_xlim((x_min, x_max))
        axs[i, 1].set_ylim((y_min, y_max))
        axs[i, 1].tick_params(axis="both", labelsize=18)

        axs[i, 2].scatter(X_HF[:, i], X_HF[:, i + 1], c="g", s=3)
        axs[i, 2].set_xlabel(f"$u_{i + 1}$", fontsize=22)
        axs[i, 2].tick_params(axis="both", labelsize=12)
        axs[i, 2].set_xlim((x_min, x_max))
        axs[i, 2].set_ylim((y_min, y_max))
        axs[i, 2].tick_params(axis="both", labelsize=18)
    axs[0, 0].legend(fontsize=15)
    # Remove x-ticks for the first 3 rows and 3 columns
    for r, c in np.ndindex(3, 3):
        axs[r, c].set_xticks([])

    # Remove y-ticks for the first 4 rows and columns 1 and 2
    for r, c in np.ndindex(4, 2):
        axs[r, c + 1].set_yticks([])


def _plot_mf_comparison_curves(
    X_LF: np.ndarray,
    X_MF: np.ndarray,
    X_HF: np.ndarray,
    x_values: np.ndarray,
    xlabel: str,
    ylabel: str,
    samples: list[int] | None = None,
    ymax: float | None = None,
    ymin: float | None = None,
    legend_loc: str = "lower right",
    figsize: tuple[float, float] = (12.5, 8),
    grid: bool = True,
) -> None:
    """
    General function to plot multi-fidelity comparison curves.

    Parameters:
    - X_LF: Low-fidelity data (2D array).
    - X_MF: Multi-fidelity data (2D array).
    - X_HF: High-fidelity data (2D array).
    - x_values: Array of x-axis values for plotting.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - samples: List of samples to plot (default: None).
    - ymax: Maximum value for the y-axis (default: None).
    - ymin: Minimum value for the y-axis (default: None).
    - legend_loc: Location for the legend in the last subplot (default: 'lower right').
    - figsize: Size of the figure (default: (12.5, 8)).
    - grid: Whether to show grid (default: True).
    """
    if samples is None:
        samples = np.random.choice(X_LF.shape[0], 4, replace=False)
        logger.info(f"Selected samples: {samples}")
    else:
        n_samples = len(samples)
        if n_samples != 4:
            logger.warning(
                f"Expected 4 samples, but got {n_samples}. "
                "Only the first 4 samples will be plotted."
            )

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.subplots_adjust(wspace=0.075, hspace=0.075)

    ymin_ = np.inf if ymin is None else ymin
    ymax_ = -np.inf if ymax is None else ymax

    # Loop over the 2x2 grid of subplots
    for i, (r, c) in enumerate(np.ndindex(2, 2)):
        j = samples[i]

        # Plot low, multi, and high-fidelity curves
        axs[r, c].plot(x_values, X_MF[j], label="Multi-Fidelity")
        axs[r, c].plot(x_values, X_LF[j], label="Low-Fidelity")
        axs[r, c].plot(x_values, X_HF[j], label="High-Fidelity")

        axs[r, c].grid(grid)

        # Configure axis labels
        if r == 1:
            axs[r, c].set_xlabel(xlabel, fontsize=18)
        if c == 0:
            axs[r, c].set_ylabel(ylabel, rotation=0, fontsize=18, labelpad=20)

        # Hide tick labels for the appropriate axes
        if r == 0:
            axs[r, c].set_xticklabels([])
        if c == 1:
            axs[r, c].set_yticklabels([])

        # Calculate the minimum value for consistent y-axis limits
        if ymin is None:
            ymin_ = min(ymin_, 1.2 * np.min([X_LF[j], X_HF[j]]))
        if ymax is None:
            ymax_ = max(ymax_, 1.2 * np.max([X_LF[j], X_HF[j]]))

    # Set consistent y-limits for all subplots
    for ax in axs.flat:
        ax.set_ylim([ymin_, ymax_])

    # Add legend in the bottom-right subplot
    axs[1, 1].legend(loc=legend_loc, fontsize=14)

    plt.show()
