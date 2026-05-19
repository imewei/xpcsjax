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

    For a two-time correlation matrix C2(t1, t2), only the upper triangle
    (t2 >= t1) contains independent data. This extracts those elements
    in row-major order for residual computation.

    Args:
        matrix: Square matrix of shape (N, N)
        include_diagonal: Whether to include diagonal elements

    Returns:
        1D array of upper-triangle values
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
    """Reconstruct symmetric matrix from upper-triangle values.

    Args:
        flat: 1D array of upper-triangle values
        n: Matrix size
        include_diagonal: Whether flat includes diagonal

    Returns:
        Symmetric matrix of shape (n, n)
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
    """Compute weight array for NLSQ fitting.

    Args:
        c2_data: Correlation data, shape (N, N)
        method: Weight method:
            - 'uniform': Equal weights (1.0)
            - 'inverse_variance': 1/sigma² from provided sigma
            - 'data_amplitude': 1/|data| for heteroscedastic data
        sigma: Standard deviation array for 'inverse_variance' method
        exclude_diagonal: Zero out diagonal weights (diagonal often noisy)

    Returns:
        Weight array of shape (N, N) where weights = 1/sigma²
    """
    if method == "uniform":
        weights = np.ones_like(c2_data)

    elif method == "inverse_variance":
        if sigma is None:
            raise ValueError("sigma required for inverse_variance weighting")
        if sigma.shape != c2_data.shape:
            raise ValueError(
                f"sigma shape {sigma.shape} doesn't match data shape {c2_data.shape}"
            )
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

    Flattens data and weights into 1D arrays suitable for
    nlsq.CurveFit (JAX-native trust-region), optionally using only the
    upper triangle of the symmetric matrix.

    Args:
        c2_data: Correlation matrix, shape (N, N)
        weights: Optional weight matrix, shape (N, N). Defaults to uniform.
        use_upper_triangle: Use only upper triangle (recommended for symmetry)
        exclude_diagonal: Exclude diagonal from fit

    Returns:
        Tuple of:
        - data_flat: 1D flattened data
        - weights_flat: 1D flattened weights (sqrt for residual scaling)
        - n_data: Number of data points
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
    """Compute degrees of freedom for chi-squared calculation.

    Args:
        n_data: Number of data points with non-zero weight
        n_params: Number of varying parameters

    Returns:
        Degrees of freedom (n_data - n_params), minimum 1
    """
    dof = max(n_data - n_params, 1)
    if n_data <= n_params:
        logger.warning(
            "Underdetermined system: %d data points, %d parameters", n_data, n_params
        )
    return dof
