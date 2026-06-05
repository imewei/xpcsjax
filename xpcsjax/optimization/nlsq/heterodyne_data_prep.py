"""Data preparation for NLSQ fitting.

Converts correlation matrices and weights into the flat arrays that
nlsq.CurveFit (JAX-native trust-region) expects, and constructs appropriate
weight arrays from data statistics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def flatten_upper_triangle(
    matrix: np.ndarray,
    include_diagonal: bool = True,
) -> np.ndarray:
    """Flatten the upper triangle of a symmetric matrix.

    For a two-time correlation matrix ``C2(t1, t2)``, only the upper triangle
    (``t2 >= t1``) contains independent data. This extracts those elements in
    row-major order for residual computation.

    Parameters
    ----------
    matrix
        Square matrix of shape ``(N, N)``.
    include_diagonal
        Whether to include the diagonal elements (``k=0`` vs ``k=1``).

    Returns
    -------
    numpy.ndarray
        1-D array of the upper-triangle values.

    Raises
    ------
    ValueError
        If *matrix* is not square (2-D with equal dimensions).
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {matrix.shape}")

    n = matrix.shape[0]
    if include_diagonal:
        indices = np.triu_indices(n, k=0)
    else:
        indices = np.triu_indices(n, k=1)

    return matrix[indices]


def unflatten_upper_triangle(
    flat: np.ndarray,
    n: int,
    include_diagonal: bool = True,
) -> np.ndarray:
    """Reconstruct a symmetric matrix from its upper-triangle values.

    Inverse of :func:`flatten_upper_triangle`: places *flat* back into the
    upper triangle and mirrors it into the lower triangle.

    Parameters
    ----------
    flat
        1-D array of upper-triangle values.
    n
        Side length of the reconstructed square matrix.
    include_diagonal
        Whether *flat* includes the diagonal (must match how it was flattened).

    Returns
    -------
    numpy.ndarray
        Symmetric matrix of shape ``(n, n)``.

    Raises
    ------
    ValueError
        If ``len(flat)`` does not match the expected upper-triangle count for
        *n* and *include_diagonal*.
    """
    matrix = np.zeros((n, n))
    if include_diagonal:
        indices = np.triu_indices(n, k=0)
    else:
        indices = np.triu_indices(n, k=1)

    expected_len = len(indices[0])
    if len(flat) != expected_len:
        raise ValueError(
            f"Expected {expected_len} values for n={n} "
            f"(include_diagonal={include_diagonal}), got {len(flat)}"
        )

    matrix[indices] = flat
    # Mirror to lower triangle
    matrix = matrix + matrix.T
    if include_diagonal:
        np.fill_diagonal(matrix, np.diag(matrix) / 2)

    return matrix


def compute_weights(
    c2_data: np.ndarray,
    method: str = "uniform",
    sigma: np.ndarray | None = None,
    exclude_diagonal: bool = False,
) -> np.ndarray:
    """Compute a weight array for NLSQ fitting.

    Parameters
    ----------
    c2_data
        Correlation data of shape ``(N, N)``.
    method
        Weighting scheme:

        - ``"uniform"`` — equal weights (1.0).
        - ``"inverse_variance"`` — ``1 / sigma**2`` from the provided *sigma*.
        - ``"data_amplitude"`` — ``1 / |data|`` for heteroscedastic data.
    sigma
        Standard-deviation array, required for the ``"inverse_variance"``
        method and ignored otherwise.
    exclude_diagonal
        Zero out the diagonal weights (the diagonal is often noisy).

    Returns
    -------
    numpy.ndarray
        Weight array of shape ``(N, N)``.

    Raises
    ------
    ValueError
        If *sigma* is missing or mis-shaped for ``"inverse_variance"``, or if
        *method* is not one of the recognised schemes.
    """
    if method == "uniform":
        weights = np.ones_like(c2_data)

    elif method == "inverse_variance":
        if sigma is None:
            raise ValueError("sigma required for inverse_variance weighting")
        if sigma.shape != c2_data.shape:
            raise ValueError(f"sigma shape {sigma.shape} doesn't match data shape {c2_data.shape}")
        # Clamp sigma to avoid division by zero
        sigma_safe = np.maximum(np.abs(sigma), 1e-30)
        weights = 1.0 / (sigma_safe**2)

    elif method == "data_amplitude":
        amplitude = np.maximum(np.abs(c2_data), 1e-30)
        weights = 1.0 / amplitude

    else:
        raise ValueError(f"Unknown weight method: {method!r}")

    if exclude_diagonal:
        np.fill_diagonal(weights, 0.0)

    return weights


def prepare_fit_data(
    c2_data: np.ndarray,
    weights: np.ndarray | None = None,
    use_upper_triangle: bool = True,
    exclude_diagonal: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Prepare correlation data and weights for least-squares fitting.

    Flattens the data and weights into 1-D arrays suitable for
    ``nlsq.CurveFit`` (the JAX-native trust-region solver), optionally using
    only the upper triangle of the symmetric matrix. Weights are returned as
    their square roots so that ``sum(residual**2)`` equals the weighted
    sum-of-squares ``sum(w * (model - data)**2)``.

    Parameters
    ----------
    c2_data
        Correlation matrix of shape ``(N, N)``.
    weights
        Optional weight matrix of shape ``(N, N)``; defaults to uniform.
    use_upper_triangle
        Use only the upper triangle (recommended, exploits symmetry).
    exclude_diagonal
        Exclude the diagonal from the fit by zeroing its weights.

    Returns
    -------
    data_flat : numpy.ndarray
        1-D flattened data.
    weights_flat : numpy.ndarray
        1-D flattened square-root weights for residual scaling.
    n_data : int
        Number of data points with non-zero weight.
    """
    if weights is None:
        weights = np.ones_like(c2_data)

    if exclude_diagonal:
        weights = weights.copy()
        np.fill_diagonal(weights, 0.0)

    if use_upper_triangle:
        data_flat = flatten_upper_triangle(c2_data)
        weights_flat = flatten_upper_triangle(weights)
    else:
        data_flat = c2_data.ravel()
        weights_flat = weights.ravel()

    # Convert weights to sqrt for residual scaling:
    # residual_i = sqrt(w_i) * (model_i - data_i)
    # so that sum(residual²) = sum(w * (model - data)²)
    sqrt_weights = np.sqrt(np.maximum(weights_flat, 0.0))

    n_data = int(np.sum(sqrt_weights > 0))

    logger.debug(
        "Prepared fit data: %d points (%d non-zero weight) from (%d, %d) matrix",
        len(data_flat),
        n_data,
        c2_data.shape[0],
        c2_data.shape[1],
    )

    return data_flat, sqrt_weights, n_data


def compute_degrees_of_freedom(
    n_data: int,
    n_params: int,
) -> int:
    """Compute the degrees of freedom for a chi-squared calculation.

    Parameters
    ----------
    n_data
        Number of data points with non-zero weight.
    n_params
        Number of varying parameters.

    Returns
    -------
    int
        ``n_data - n_params``, clamped to a minimum of 1. An underdetermined
        system (``n_data <= n_params``) is logged as a warning.
    """
    dof = max(n_data - n_params, 1)
    if n_data <= n_params:
        logger.warning("Underdetermined system: %d data points, %d parameters", n_data, n_params)
    return dof


def far_lag_noise_variance(c2_data: np.ndarray) -> float:
    """Estimate the photon-noise variance from the far-lag tail of C2.

    For large lag ``|t1 - t2|`` the correlation has fully decayed to its
    baseline, so the residual scatter there is dominated by measurement noise
    rather than dynamics. The variance of those far-lag values is therefore a
    data-driven estimate of ``σ²_noise``. Pools the far-lag entries across all
    angles when ``c2_data`` is 3-D ``(n_phi, n_time, n_time)``.

    Mirrors the per-angle estimate in
    :func:`heterodyne_core._compute_per_angle_chi2` so the single-angle and
    joint multi-angle paths share one noise convention.

    Parameters
    ----------
    c2_data
        Per-angle C2 matrix ``(n_time, n_time)`` or batched
        ``(n_phi, n_time, n_time)``. Far-lag entries are pooled across angles
        in the batched case.

    Returns
    -------
    float
        ``var(far_lag_values)``, or ``0.0`` when too few far-lag points exist.

    See Also
    --------
    noise_normalized_reduced_chi2 : Consumes this estimate to rescale chi-squared.
    """
    c2_np = np.asarray(c2_data, dtype=np.float64)
    n_time = c2_np.shape[-1]
    row_idx = np.arange(n_time)
    lag_mat = np.abs(row_idx[:, None] - row_idx[None, :])
    far_mask = lag_mat >= n_time // 2
    far_vals = c2_np[..., far_mask].ravel()
    return float(np.var(far_vals)) if far_vals.size > 1 else 0.0


def noise_normalized_reduced_chi2(
    ssr: float,
    c2_data: np.ndarray,
    n_data_valid: int,
    n_params: int,
) -> float:
    """Noise-normalised reduced chi-squared targeting ≈ 1.0 for a good fit.

    The raw least-squares ``SSR / dof`` collapses to ``MSE ≪ 1`` on normalised
    C2 data (C2 ~ 1, residuals ~ 5%), which is statistically meaningless as a
    goodness-of-fit. Dividing the SSR additionally by an estimated far-lag
    photon-noise variance restores the conventional ``χ²_red ≈ 1`` scale. This
    is the same correction the single-angle / per-angle / CMA-ES heterodyne
    paths already apply (see ``heterodyne_core._compute_per_angle_chi2`` and the
    ``chi2_corrected`` block in ``_run_nlsq_with_cmaes_escape``); centralising it
    here keeps every joint path (averaged, fourier, constant) consistent.

    Falls back to plain MSE (``SSR / dof``) when the noise estimate is
    degenerate, matching the fallback used elsewhere.

    Parameters
    ----------
    ssr
        Data-only sum of squared residuals.
    c2_data
        Per-angle C2 matrix or batched ``(n_phi, n_time, n_time)``; used only
        to estimate ``sigma**2_noise``.
    n_data_valid
        Number of residuals actually fit (off-diagonal, ``t=0`` boundary
        excluded) — i.e. ``len(data_only_residual)``.
    n_params
        Number of fitted parameters.

    Returns
    -------
    float
        Noise-normalised reduced chi-squared, with an MSE fallback when the
        noise estimate is degenerate.

    See Also
    --------
    far_lag_noise_variance : Supplies the ``sigma**2_noise`` estimate used here.
    """
    n_dof = max(int(n_data_valid) - int(n_params), 1)
    sigma2_noise = far_lag_noise_variance(c2_data)
    if sigma2_noise > 1e-12:
        return float(ssr) / (sigma2_noise * n_dof)
    return float(ssr) / n_dof
