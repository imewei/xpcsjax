"""Jacobian computation utilities for NLSQ optimization.

This module extracts Jacobian-related functions from nlsq_wrapper.py
to reduce file size and improve maintainability.

Extracted from nlsq_wrapper.py as part of technical debt remediation (Dec 2025).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def compute_jacobian_stats(
    residual_fn: Callable[..., Any],
    x_subset: np.ndarray,
    params: np.ndarray,
    scaling_factor: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute Jacobian statistics for convergence diagnostics.

    Computes the Jacobian matrix and derives:
    - JTJ (Jacobian transpose times Jacobian) for Hessian approximation
    - Column norms for parameter sensitivity analysis

    Parameters
    ----------
    residual_fn : Callable
        Residual function to differentiate.
    x_subset : np.ndarray
        Subset of x data for Jacobian computation.
    params : np.ndarray
        Current parameter values.
    scaling_factor : float
        Scaling factor for JTJ computation.

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        (JTJ matrix, column norms) or (None, None) on failure.
    """
    try:
        params_jnp = jnp.asarray(params)
        if hasattr(residual_fn, "jax_residual"):

            def residual_vector(p):
                return jnp.asarray(residual_fn.jax_residual(jnp.asarray(p))).reshape(-1)

        else:

            def residual_vector(p):
                return jnp.asarray(residual_fn(x_subset, *tuple(p))).reshape(-1)

        # Use jacfwd (JVP-based): O(n × cost_f) vs jacrev's O(m × cost_f).
        # For XPCS m >> n (e.g., 20K residuals, 9 params), jacfwd is ~260x faster.
        jac = jax.jacfwd(residual_vector)(params_jnp)
        jac_np = np.asarray(jac)

        # Performance Optimization (Spec 001 - FR-010, T048): Check condition number
        # to determine optimal J^T J computation method.
        # For ill-conditioned Jacobians (cond > 1e6), use QR-based computation.
        try:
            cond_number = np.linalg.cond(jac_np)
        except np.linalg.LinAlgError:
            cond_number = np.inf

        if cond_number > 1e6:
            # Performance Optimization (Spec 001 - FR-010, T049): QR-based J^T J
            # For ill-conditioned Jacobians, J^T J = R^T R is more numerically stable.
            # QR decomposition: J = Q @ R where Q is orthogonal, R is upper triangular.
            Q, R = np.linalg.qr(jac_np)
            jtj = R.T @ R * scaling_factor
        else:
            # Standard computation for well-conditioned Jacobians
            jtj = jac_np.T @ jac_np * scaling_factor

        col_norms = np.linalg.norm(jac_np, axis=0) * np.sqrt(scaling_factor)
        return jtj, col_norms
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None, None


def compute_jacobian_condition_number(
    residual_fn: Callable[..., Any],
    x_subset: np.ndarray,
    params: np.ndarray,
) -> float | None:
    """Compute condition number of Jacobian matrix.

    The condition number indicates how sensitive the optimization
    is to parameter perturbations. High values (>1e6) suggest
    ill-conditioning.

    Parameters
    ----------
    residual_fn : Callable
        Residual function to differentiate.
    x_subset : np.ndarray
        Subset of x data for Jacobian computation.
    params : np.ndarray
        Current parameter values.

    Returns
    -------
    float | None
        Condition number or None on failure.
    """
    try:
        params_jnp = jnp.asarray(params)
        if hasattr(residual_fn, "jax_residual"):

            def residual_vector(p):
                return jnp.asarray(residual_fn.jax_residual(jnp.asarray(p))).reshape(-1)

        else:

            def residual_vector(p):
                return jnp.asarray(residual_fn(x_subset, *tuple(p))).reshape(-1)

        # Use jacfwd (JVP-based): O(n × cost_f) vs jacrev's O(m × cost_f).
        # For XPCS m >> n (e.g., 20K residuals, 9 params), jacfwd is ~260x faster.
        jac = jax.jacfwd(residual_vector)(params_jnp)
        jac_np = np.asarray(jac)
        return float(np.linalg.cond(jac_np))
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None


def analyze_parameter_sensitivity(
    residual_fn: Callable[..., Any],
    x_subset: np.ndarray,
    params: np.ndarray,
    param_names: list[str],
) -> dict[str, float]:
    """Analyze parameter sensitivity from Jacobian column norms.

    Higher column norms indicate parameters that have more influence
    on the residuals.

    Parameters
    ----------
    residual_fn : Callable
        Residual function to differentiate.
    x_subset : np.ndarray
        Subset of x data for Jacobian computation.
    params : np.ndarray
        Current parameter values.
    param_names : list[str]
        Parameter names for labeling.

    Returns
    -------
    dict[str, float]
        Mapping from parameter name to sensitivity (normalized 0-1).
    """
    _, col_norms = compute_jacobian_stats(residual_fn, x_subset, params, 1.0)
    if col_norms is None:
        return {}

    # Normalize to 0-1 range
    max_norm = np.max(col_norms)
    if max_norm > 0:
        normalized = col_norms / max_norm
    else:
        normalized = np.zeros_like(col_norms)

    return {
        name: float(norm) for name, norm in zip(param_names, normalized, strict=False)
    }


def estimate_gradient_noise(
    residual_fn: Callable[..., Any],
    x_subset: np.ndarray,
    params: np.ndarray,
    n_samples: int = 5,
    perturbation: float = 1e-6,
    seed: int = 42,
) -> float | None:
    """Estimate gradient noise from multiple Jacobian computations.

    Computes Jacobian multiple times with small perturbations to
    estimate numerical noise in gradient computation.

    Parameters
    ----------
    residual_fn : Callable
        Residual function to differentiate.
    x_subset : np.ndarray
        Subset of x data for Jacobian computation.
    params : np.ndarray
        Current parameter values.
    n_samples : int
        Number of perturbed samples.
    perturbation : float
        Relative perturbation size.

    Returns
    -------
    float | None
        Estimated gradient noise (coefficient of variation) or None on failure.
    """
    try:
        params_base = np.asarray(params, dtype=float)
        jacobians = []
        rng = np.random.default_rng(seed=seed)

        # Define residual_vector once outside the loop (branch condition is loop-invariant)
        if hasattr(residual_fn, "jax_residual"):

            def residual_vector(p):
                return jnp.asarray(residual_fn.jax_residual(jnp.asarray(p))).reshape(-1)

        else:

            def residual_vector(p):
                return jnp.asarray(residual_fn(x_subset, *tuple(p))).reshape(-1)

        for _ in range(n_samples):
            # Add small perturbation
            noise = (
                rng.standard_normal(len(params_base))
                * perturbation
                * np.abs(params_base)
            )
            params_perturbed = params_base + noise

            params_jnp = jnp.asarray(params_perturbed)

            # Use jacfwd (JVP-based): O(n × cost_f), faster for m >> n
            jac = jax.jacfwd(residual_vector)(params_jnp)
            jacobians.append(np.asarray(jac))

        # Compute coefficient of variation across samples
        jac_stack = np.stack(jacobians, axis=0)
        jac_mean = np.mean(jac_stack, axis=0)
        jac_std = np.std(jac_stack, axis=0)

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(np.abs(jac_mean) > 1e-10, jac_std / np.abs(jac_mean), 0.0)

        return float(np.median(cv))
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None
