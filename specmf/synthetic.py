"""Synthetic data generation utilities for multi-fidelity examples."""

import numpy as np
from sklearn.datasets import make_circles, make_s_curve


# ============================================================================
#   2D SYNTHETIC DATA (for figure-readme.ipynb)
# ============================================================================


def affine_transform(data, apply_stretch=True, apply_shift=True):
    """
    Applies an affine transformation to input data.

    Parameters:
    - data (np.ndarray): Input data of shape (n_samples, 2).
    - apply_stretch (bool, optional): If True, applies a 2x stretch along
      `y=x` and a 0.8x stretch along `y=-x`.
    - apply_shift (bool, optional): If True, applies a shift of 0.25 along both
      `x` and `y` axes.

    Returns:
    - np.ndarray: Transformed data
    """
    new_data = data.copy()
    if apply_stretch:
        P = (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]])
        D = np.array([[2, 0], [0, 0.8]])
        M = P @ D @ P.T
        new_data = (M @ data.T).T
    if apply_shift:
        shift = np.array([0.25, 0.25])
        new_data += shift
    return new_data


def generate_circles_2d(
    n_samples=1000,
    noise_scale_lf=0.015,
    noise_scale_hf=0.01,
    random_state=42,
):
    """
    Generate low-fidelity and high-fidelity 2D circles datasets.

    The high-fidelity data has an affine transformation applied.

    Parameters:
    - n_samples (int): Number of samples to generate
    - noise_scale_lf (float): Noise scale for low-fidelity data
    - noise_scale_hf (float): Noise scale for high-fidelity data
    - random_state (int): Random seed for reproducibility

    Returns:
    - lf_data (np.ndarray): Low-fidelity data of shape (n_samples, 2)
    - hf_data (np.ndarray): High-fidelity data of shape (n_samples, 2)
    """
    np.random.seed(random_state)

    # Generate base data once without noise
    base_data, _ = make_circles(n_samples=n_samples, factor=0.5, noise=0.0)

    # Add different levels of noise to create low- and high-fidelity data
    lf_noise = np.random.normal(scale=noise_scale_lf, size=base_data.shape)
    hf_noise = np.random.normal(scale=noise_scale_hf, size=base_data.shape)

    lf_data = base_data + lf_noise
    hf_data = base_data + hf_noise
    hf_data = affine_transform(hf_data)

    return lf_data, hf_data


# ============================================================================
#   3D SYNTHETIC DATA (for figure-presentation.ipynb)
# ============================================================================


def make_s_cluster(n=500, noise=0.01, random_state=0, return_t=False):
    """
    3D S-shaped cluster based on sklearn's make_s_curve.
    We keep the parameter t so we can unfold the S in a
    smooth, monotone way for the HF data.
    """
    X, t = make_s_curve(n_samples=n, noise=noise, random_state=random_state)
    # Scale and translate to taste
    X = np.array([1.0, 0.35, 1.0]) * X + np.array([0.0, 2.5, 5.0])

    if return_t:
        return X, t
    return X


def make_torus_cluster(n=1000, R=3.0, r=0.35, noise=0.01, random_state=0):
    """
    3D torus cluster.

    R: major radius (distance from center of hole to center of tube)
    r: minor radius (radius of tube)
    """
    rng = np.random.RandomState(random_state)
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)

    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    X = np.column_stack([x, y, z])
    X += noise * rng.randn(n, 3)

    X[:, 0] *= 1.5
    X[:, 1] *= 0.6

    return X


def make_half_moon_cluster(
    n=500,
    length=8.0,
    r_max=0.5,
    center=(6.0, 5.0, 1.0),
    random_state=0,
):
    """
    Generate a straight 'cylinder-like' point cloud.

    - Medial axis is a segment along x of total length `length`,
      centered at `center`.
    - Around the medial axis, add uniform noise in y and z in
      [-r_max, r_max].

    """
    rng = np.random.RandomState(random_state)
    cx, cy, cz = center

    # Coordinate along the medial axis (x-direction), centered at cx
    s = rng.uniform(-0.5, 0.5, n) * length  # axis parameter in [-L/2, L/2]

    # Straight medial axis
    x = cx + s

    # Uniform noise around the axis in y, z
    # (change to [0, r_max] if you want only one side)
    y = cy + rng.uniform(-r_max, r_max, n)
    z = cz + rng.uniform(-r_max, r_max, n)

    X = np.column_stack([x, y, z])

    return X


# ======================================================
#   UNFOLDING TRANSFORMS
# ======================================================


def unfold_s_curve(X, t, length=8.0):
    """
    Unfold the S-curve into a straight vertical curtain:
    - Preserve ordering via parameter t.
    - Keep the center of mass fixed in x,z.
    - Keep y (height) as in the original so the 'vertical'
      structure at the beginning stays vertical.

    Result: a straight band aligned mainly along z, centered
    where the S originally was.
    """
    center = X.mean(axis=0)  # (cx, cy, cz)

    # Normalize t to [-0.5, 0.5]
    t_norm = (t - t.min()) / (t.max() - t.min()) - 0.5

    # Map to a straight line in z, keep y from original, fix x at center
    x_new = np.full_like(t_norm, center[0])
    y_new = X[:, 1]  # keep vertical structure
    z_new = center[2] - length * t_norm

    return np.column_stack([x_new, y_new, z_new])


def bend_half_moon(X, center=(6.0, 5.0, 1.0)):
    """
    Bend a straight cylinder into a 'half-moon'-like shape by
    squaring the y-coordinate w.r.t. the center.

    For each point:
        d = X - center
        d_y_new = d_y ** 2
        X_bent = center + (d_x, d_y_new, d_z)

    This gives a smooth, deterministic mapping LF -> HF.
    """
    center = np.asarray(center, dtype=float)
    d = X - center

    # Square the second component (y) w.r.t. the center
    d[:, 2] = 0.5 * d[:, 0] ** 2

    X_bent = d + center
    return X_bent


# ======================================================
#   GENERATE LF + HF (3D)
# ======================================================


def generate_lf_hf_example(
    n_s=600, n_torus=1500, n_halfmoon=800, random_state=0
):
    """
    Generate low-fidelity (LF) and transformed high-fidelity (HF)
    datasets with 3 clusters: S-shape, torus, and half-moon.

    LF:
      - S-curve (make_s_curve)
      - Torus
      - Half-moon

    HF:
      - Torus: stretched in y, shrunk in x
      - S-curve: unfolded into a vertical straight 'curtain'
      - Half-moon: opened into a horizontal straight segment

    Returns:
    - lf_data (np.ndarray): Low-fidelity data of shape (n_total, 3)
    - hf_data (np.ndarray): High-fidelity data of shape (n_total, 3)
    - labels (np.ndarray): Cluster labels for each point
    """
    # --- HF clusters ---
    s_hf, t_s = make_s_cluster(n=n_s, random_state=random_state, return_t=True)
    torus_hf = make_torus_cluster(n=n_torus, random_state=random_state + 1)
    half_lf = make_half_moon_cluster(
        n=n_halfmoon, random_state=random_state + 2
    )
    half_hf = bend_half_moon(half_lf)

    hf_data = np.vstack([s_hf, torus_hf, half_hf])
    labels = np.hstack([
        np.zeros(len(s_hf), dtype=int),
        np.ones(len(torus_hf), dtype=int),
        2 * np.ones(len(half_hf), dtype=int),
    ])

    # --- LF transformations ---

    # 1) S-curve: unfold using parameter t
    s_lf = unfold_s_curve(s_hf, t_s, length=8.0)

    # 2) Torus: stretch x, shrink y
    torus_lf = torus_hf.copy()
    torus_center = torus_hf.mean(axis=0)
    torus_lf = torus_lf - torus_center
    torus_lf[:, 0] /= 1.5
    torus_lf[:, 1] /= 0.6
    torus_lf += torus_center

    # 3) Half-moon: keep as is (already in LF form)

    lf_data = np.vstack([s_lf, torus_lf, half_lf])

    return lf_data, hf_data, labels
