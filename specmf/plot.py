import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
    }
)


# General plotting functions
def plot_distributions(
    e_LF: np.ndarray,
    e_MF: np.ndarray,
    bins_LF: int = 50,
    bins_MF: int = 50,
    mask: Optional[np.ndarray] = None,
    return_axs: bool = False,
) -> None:
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

    if return_axs:
        return axs


def plot_spectrum(eigvals: np.ndarray, n: int) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(np.arange(1, n + 1), np.abs(eigvals[:n]), "o-", c="k", markersize=5)
    ax.set_yscale("log")
    ax.set_xticks(range(1, n + 1, 5))
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlabel("$m$", fontsize=20)
    ax.set_ylabel(r"$\lambda_m$", fontsize=20, rotation=0, labelpad=20)
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()


def plot_cluster_size_hist(labels: np.ndarray) -> None:
    """
    Plot histogram of cluster sizes.
    """
    N = np.max(labels) + 1
    clus_sizes = [np.size(labels[labels == c]) for c in range(N)]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(np.sort(clus_sizes), bins=20)
    ax.set_xlabel("Cluster size", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title("Clusters size histogram", fontsize=18)


# Data plotting functions
def plot_data(X_LF: np.ndarray, X_HF: np.ndarray, dataset_name: str, **kwargs) -> None:
    """
    Plot data based on the dataset name.

    Parameters:
    - X_LF: Low-fidelity data.
    - X_HF: High-fidelity data.
    - dataset_name: Name of the dataset.
    """
    plotters = {
        "inclusion-field": _plot_lf_hf_fields_samples,
        "darcy-flow": _plot_lf_hf_fields_samples,
        "inclusion-qoi": _plot_lf_hf_qoi_samples,
        "beam": _plot_lf_hf_curves_samples,
        "cavity": _plot_lf_hf_curves_samples,
        "burgers": _plot_lf_hf_curves_samples,
    }

    if dataset_name not in plotters:
        raise ValueError(
            f"Invalid dataset name. Expected one of {plotters.keys()}, "
            f"got {dataset_name} instead."
        )

    plotters[dataset_name](X_LF, X_HF, **kwargs)
    plt.show()


def _plot_lf_hf_fields_samples(
    X_LF: np.ndarray, X_HF: np.ndarray, n_samples: int = 5, levels: int = 10, **kwargs
) -> None:
    n_points, n_dim = X_LF.shape
    n_res = int(np.sqrt(n_dim))
    for i in range(n_samples):
        j = np.random.randint(0, n_points)
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.subplots_adjust(wspace=0.15)
        axs[0].set_title("Low-fidelity", fontsize=20)
        axs[1].set_title("High-fidelity", fontsize=20)
        axs[2].set_title("Difference", fontsize=20)
        vmin, vmax = np.min((X_LF[j, :], X_HF[j, :])), np.max((X_LF[j, :], X_HF[j, :]))
        lf = axs[0].contourf(
            X_LF[j, :].reshape(n_res, n_res), levels=levels, vmin=vmin, vmax=vmax
        )
        hf = axs[1].contourf(
            X_HF[j, :].reshape(n_res, n_res), levels=levels, vmin=vmin, vmax=vmax
        )
        diff = axs[2].contourf(
            np.abs(X_HF[j, :].reshape(n_res, n_res) - X_LF[j, :].reshape(n_res, n_res)),
            cmap="Reds",
        )
        plt.colorbar(lf, ax=axs[0])
        plt.colorbar(hf, ax=axs[1])
        plt.colorbar(diff, ax=axs[2])


def _plot_lf_hf_curves_samples(
    X_LF: np.ndarray, X_HF: np.ndarray, n_samples: int = 5
) -> None:
    n_points, _ = X_LF.shape
    fig, axs = plt.subplots(n_samples, 1, figsize=(5, 4 * n_samples))
    fig.subplots_adjust(wspace=0.1)

    for i in range(n_samples):
        j = np.random.randint(0, n_points)
        vmin, vmax = 1.2 * np.min((X_LF[j, :], X_HF[j, :])), 1.2 * np.max(
            (X_LF[j, :], X_HF[j, :])
        )
        x_ = np.linspace(0, 1, X_LF[j, :].shape[0])
        axs[i].plot(x_, X_LF[j, :], label="Low-Fidelity")
        axs[i].plot(x_, X_HF[j, :], label="High-Fidelity")
        axs[i].set_ylim([vmin, vmax])
        axs[i].legend()
        axs[i].grid("on")


def _plot_lf_hf_qoi_samples(X_LF: np.ndarray, X_HF: np.ndarray, **kwargs) -> None:
    # Plot clustered dataset and centorids
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    fig.subplots_adjust(wspace=0.3)
    n_planes = X_LF.shape[1] - 1

    for i in range(n_planes):
        y_max, y_min = 1.2 * np.max((X_LF[:, i + 1], X_HF[:, i + 1])), 1.2 * np.min(
            (X_LF[:, i + 1], X_HF[:, i + 1])
        )
        x_max, x_min = 1.2 * np.max((X_LF[:, i], X_HF[:, i])), 1.2 * np.min(
            (X_LF[:, i], X_HF[:, i])
        )

        axs[0, i].scatter(X_LF[:, i], X_LF[:, i + 1], s=3)
        axs[0, i].set_xlabel("$u_{}$".format(i + 1), fontsize=24)
        axs[0, i].set_ylabel(
            "$u_{}$".format(i + 2), fontsize=24, rotation=0, labelpad=15
        )
        axs[0, i].tick_params(axis="both", labelsize=16)
        axs[0, i].set_xlim((x_min, x_max))
        axs[0, i].set_ylim((y_min, y_max))

        axs[1, i].scatter(X_HF[:, i], X_HF[:, i + 1], s=3, c="r")
        axs[1, i].set_xlabel(
            "$u_{}$".format(i + 1),
            fontsize=24,
        )
        axs[1, i].set_ylabel(
            "$u_{}$".format(i + 2),
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

    plotters = {
        "inclusion-field": _plot_mf_comparison_field,
        "darcy-flow": _plot_mf_comparison_field,
        "inclusion-qoi": _plot_mf_comparison_qoi,
        "beam": _plot_mf_comparison_curves,
        "cavity": _plot_mf_comparison_curves,
        "burgers": _plot_mf_comparison_curves,
    }

    if dataset_name not in plotters:
        raise ValueError(
            f"Invalid dataset name. Expected one of {plotters.keys()}, "
            f"got {dataset_name} instead."
        )

    plotters[dataset_name](X_LF, X_MF, X_HF, **kwargs)
    plt.show()


def _plot_mf_comparison_field(
    X_LF: np.ndarray, X_MF: np.ndarray, X_HF: np.ndarray, n_samples: int = 4, **kwargs
) -> None:
    from matplotlib.ticker import MaxNLocator

    n_points, n_dim = X_LF.shape
    n_res = int(np.sqrt(n_dim))

    fig, axs = plt.subplots(n_samples, 4, figsize=(20, 4.25 * n_samples))
    fig.subplots_adjust(
        wspace=0.1,
    )
    axs[0, 0].set_title("Low-Fidelity", fontsize=24, pad=20)
    axs[0, 1].set_title("Low-Fidelity Error", fontsize=24, pad=20)
    axs[0, 2].set_title("Multi-Fidelity", fontsize=24, pad=20)
    axs[0, 3].set_title("Multi-Fidelity Error", fontsize=24, pad=20)

    for i in range(n_samples):
        j = np.random.randint(n_points)
        vmin, vmax = np.min((X_LF[j, :], X_HF[j, :])), np.max((X_LF[j, :], X_HF[j, :]))
        vmax_diff = np.max(
            (np.abs(X_HF[j, :] - X_LF[j, :]), np.abs(X_MF[j, :] - X_LF[j, :]))
        )

        lf = axs[i, 0].contourf(
            X_LF[j, :].reshape(n_res, n_res), vmin=vmin, vmax=vmax, levels=10
        )
        lf_diff = axs[i, 1].contourf(
            np.abs(X_HF[j, :] - X_LF[j, :]).reshape(n_res, n_res),
            vmin=0,
            vmax=vmax_diff,
            cmap="Reds",
            levels=10,
        )
        mf = axs[i, 2].contourf(
            X_MF[j, :].reshape(n_res, n_res), vmin=vmin, vmax=vmax, levels=10
        )
        mf_diff = axs[i, 3].contourf(
            np.abs(X_HF[j, :] - X_MF[j, :]).reshape(n_res, n_res),
            vmin=0,
            vmax=vmax_diff,
            cmap="Reds",
            levels=10,
        )

        for ax in axs[i, :]:
            ax.set_xticks([])
            ax.set_yticks([])

        cbar = plt.colorbar(lf, ax=axs[i, 0])
        cbar.ax.tick_params(labelsize=14)
        cbar.locator = MaxNLocator(nbins=5)
        cbar.update_ticks()
        cbar = plt.colorbar(lf_diff, ax=axs[i, 1])
        cbar.ax.tick_params(labelsize=14)
        cbar.locator = MaxNLocator(nbins=5)
        cbar.update_ticks()
        cbar = plt.colorbar(mf, ax=axs[i, 2])
        cbar.ax.tick_params(labelsize=14)
        cbar.locator = MaxNLocator(nbins=5)
        cbar.update_ticks()
        cbar = plt.colorbar(mf_diff, ax=axs[i, 3])
        cbar.ax.tick_params(labelsize=14)
        cbar.locator = MaxNLocator(nbins=5)
        cbar.update_ticks()


def _plot_mf_comparison_qoi(
    X_LF: np.ndarray,
    X_MF: np.ndarray,
    X_HF: np.ndarray,
    dPhi: np.ndarray,
    inds_centroids: list,
    **kwargs,
) -> None:

    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    fig.subplots_adjust(wspace=0.4)

    axs[0, 0].set_title("Low-Fidelity", fontsize=24)
    axs[0, 1].set_title("Multi-Fidelity - mean", fontsize=24)
    axs[0, 2].set_title("Multi-Fidelity - 1 std", fontsize=24)
    axs[0, 3].set_title("High-Fidelity", fontsize=24)

    for i in range(4):
        y_min = 1.1 * np.min((X_LF[:, i + 1], X_MF[:, i + 1], X_HF[:, i + 1]))
        y_max = 1.1 * np.max((X_LF[:, i + 1], X_MF[:, i + 1], X_HF[:, i + 1]))

        x_max = 1.1 * np.max((X_LF[:, i], X_MF[:, i], X_HF[:, i]))
        x_min = 1.1 * np.min((X_LF[:, i], X_MF[:, i], X_HF[:, i]))

        axs[i, 0].scatter(X_LF[:, i], X_LF[:, i + 1], c="orange", s=3)
        axs[i, 0].scatter(
            X_LF[inds_centroids, i],
            X_LF[inds_centroids, i + 1],
            c="blue",
            s=10,
            label="Centroids",
        )
        axs[i, 0].set_xlabel("$u_{}$".format(i + 1), fontsize=22)
        axs[i, 0].set_ylabel(
            "$u_{}$".format(i + 2), fontsize=22, rotation=0, labelpad=15
        )
        axs[i, 0].tick_params(axis="both", labelsize=12)
        axs[i, 0].set_xlim((x_min, x_max))
        axs[i, 0].set_ylim((y_min, y_max))
        axs[i, 0].tick_params(axis="both", labelsize=18)

        axs[i, 1].scatter(X_MF[:, i], X_MF[:, i + 1], s=3)
        axs[i, 1].set_xlabel("$u_{}$".format(i + 1), fontsize=22)
        axs[i, 1].tick_params(axis="both", labelsize=12)
        axs[i, 1].set_xlim((x_min, x_max))
        axs[i, 1].set_ylim((y_min, y_max))
        axs[i, 1].tick_params(axis="both", labelsize=18)

        axs[i, 2].set_xlabel("$u_{}$".format(i + 1), fontsize=22)
        axs[i, 2].tick_params(axis="both", labelsize=12)
        axs[i, 2].set_xlim((x_min, x_max))
        axs[i, 2].set_ylim((y_min, y_max))
        color_std = (dPhi - dPhi.min()) / (dPhi.max() - dPhi.min())
        axs[i, 2].scatter(X_MF[:, i], X_MF[:, i + 1], s=10, c=color_std[:])
        axs[i, 2].tick_params(axis="both", labelsize=18)

        axs[i, 3].scatter(X_HF[:, i], X_HF[:, i + 1], c="r", s=3)
        axs[i, 3].set_xlabel("$u_{}$".format(i + 1), fontsize=22)
        axs[i, 3].tick_params(axis="both", labelsize=12)
        axs[i, 3].set_xlim((x_min, x_max))
        axs[i, 3].set_ylim((y_min, y_max))
        axs[i, 3].tick_params(axis="both", labelsize=18)
    axs[0, 0].legend(fontsize=15)


def _plot_mf_comparison_curves(
    X_LF: np.ndarray, X_MF: np.ndarray, X_HF: np.ndarray, n_samples: int = 5, **kwargs
) -> None:
    n_points, _ = X_LF.shape
    fig, axs = plt.subplots(n_samples, 1, figsize=(5, 4 * n_samples))
    fig.subplots_adjust(wspace=0.1)

    for i in range(n_samples):
        j = np.random.randint(0, n_points)
        vmin, vmax = 1.2 * np.min((X_LF[j, :], X_HF[j, :])), 1.2 * np.max(
            (X_LF[j, :], X_HF[j, :])
        )
        x_ = np.linspace(0, 1, X_LF[j, :].shape[0])
        axs[i].plot(x_, X_LF[j, :], label="Low-Fidelity")
        axs[i].plot(x_, X_HF[j, :], label="High-Fidelity")
        axs[i].plot(x_, X_MF[j, :], label="Multi-Fidelity")
        axs[i].set_ylim([vmin, vmax])
        axs[i].legend()
        axs[i].grid("on")
