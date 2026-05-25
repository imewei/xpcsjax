"""Gradient diagnostics and x_scale recommender for NLSQ optimization.

Diagnoses gradient-magnitude imbalance between physical parameters (e.g.
shear params vs diffusion params) and computes per-parameter ``x_scale``
values that normalise optimisation steps.

This complements :mod:`xpcsjax.optimization.nlsq.gradient_monitor`, which
detects gradient collapse at runtime. The diagnostics here are offline:
given a fitted parameter point, recommend an ``x_scale_map`` for the next
optimisation.

The Problem
-----------
Shear parameters (``gamma_dot_t0``, ``beta``, ``gamma_dot_t_offset``) can
have gradients 100x-10000x larger than diffusion parameters (``D0``,
``alpha``, ``D_offset``), causing premature convergence, missed fine-scale
features, and poor fit quality despite low chi-squared.

The Solution
------------
Compute parameter-specific ``x_scale`` values inversely proportional to
gradient magnitudes to normalise optimisation steps across all parameters.

Usage
-----
.. code-block:: python

    from xpcsjax.optimization.nlsq.gradient_diagnostics import (
        compute_optimal_x_scale,
    )

    x_scale_map = compute_optimal_x_scale(
        parameters=result.parameters,
        data=data,
        config=config,
        analysis_mode="laminar_flow",
    )
    config.config["optimization"]["nlsq"]["x_scale_map"] = x_scale_map
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


def _create_residual_function(
    data: Any,
    analysis_mode: AnalysisMode,
) -> tuple[Callable, list[str]]:
    """Create residual function for gradient computation.

    Args:
        data: Data object with phi, t1, t2, g2, q, L, dt attributes
        analysis_mode: "static_isotropic" or "laminar_flow"

    Returns:
        (residual_fn, param_names): Residual function and parameter names
    """
    from xpcsjax.core.jax_backend import compute_g1_total

    phi = jnp.asarray(data.phi)
    t1 = jnp.asarray(data.t1)
    t2 = jnp.asarray(data.t2)
    g2_exp = jnp.asarray(data.g2)
    q = float(data.q)
    L = float(data.L)
    if hasattr(data, "dt") and data.dt is not None:
        dt = float(data.dt)
    else:
        logger.warning(
            "data.dt is missing or None; using dt=1.0 for gradient diagnostics. "
            "Gradient norms will be correct only if the true frame interval is 1.0 s."
        )
        dt = 1.0

    if hasattr(data, "per_angle_scaling_solver"):
        per_angle = np.asarray(data.per_angle_scaling_solver)
        contrasts = jnp.asarray(per_angle[:, 0])
        offsets = jnp.asarray(per_angle[:, 1])
    else:
        n_phi = len(np.unique(phi))
        contrasts = jnp.ones(n_phi) * 0.5
        offsets = jnp.ones(n_phi) * 1.0

    phi_unique_sorted = jnp.array(sorted({float(p) for p in np.asarray(data.phi)}))

    if "static" in analysis_mode.lower():
        param_names = ["D0", "alpha", "D_offset"]
    else:
        param_names = [
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]

    @jax.jit
    def residual_fn(params: jnp.ndarray) -> jnp.ndarray:
        g1 = compute_g1_total(params, t1, t2, phi, q, L, dt)
        phi_idx = jnp.searchsorted(phi_unique_sorted, phi)
        contrast_per_point = contrasts[phi_idx]
        offset_per_point = offsets[phi_idx]
        g2_theory = offset_per_point + contrast_per_point * jnp.square(g1)
        residuals = (g2_theory - g2_exp).reshape(-1)
        return residuals

    return residual_fn, param_names


def compute_gradient_norms(
    parameters: dict[str, float],
    data: Any,
    config: Any,
    analysis_mode: AnalysisMode,
) -> dict[str, float]:
    """Compute gradient L2 norms for each parameter at the given point.

    Args:
        parameters: Dictionary of parameter values
        data: Data object with experimental data
        config: Configuration object
        analysis_mode: "static_isotropic" or "laminar_flow"

    Returns:
        Dictionary mapping parameter names to gradient norms
    """
    del config  # accepted for API symmetry with the public callers
    residual_fn, param_names = _create_residual_function(data, analysis_mode)

    param_array = jnp.array([float(parameters[name]) for name in param_names])

    def sse_fn(params: jnp.ndarray) -> jnp.ndarray:
        residuals = residual_fn(params)
        return jnp.sum(residuals**2)

    grad_fn = jax.grad(sse_fn)
    gradients = grad_fn(param_array)

    gradient_norms = {
        name: float(abs(grad))
        for name, grad in zip(param_names, gradients, strict=False)
    }

    return gradient_norms


def compute_optimal_x_scale(
    parameters: dict[str, float],
    data: Any,
    config: Any,
    analysis_mode: AnalysisMode,
    baseline_params: list[str] | None = None,
    safety_factor: float = 1.0,
    min_scale: float = 1e-8,
    max_scale: float = 1e2,
) -> dict[str, float]:
    """Compute optimal x_scale map based on gradient norms.

    The x_scale values are inversely proportional to gradient magnitudes,
    normalised so that baseline parameters have ``x_scale=1.0``.

    Args:
        parameters: Dictionary of parameter values
        data: Data object with experimental data
        config: Configuration object
        analysis_mode: "static_isotropic" or "laminar_flow"
        baseline_params: Parameters to use as baseline (x_scale=1.0).
            Default: ``["D0", "D_offset", "phi0"]`` (laminar) or
            ``["D0", "D_offset"]`` (static).
        safety_factor: Multiplicative safety factor (default: 1.0). Increase
            to make optimisation more conservative.
        min_scale: Minimum allowed x_scale value (prevents division by zero)
        max_scale: Maximum allowed x_scale value (prevents extreme values)

    Returns:
        Dictionary mapping parameter names to x_scale values
    """
    if baseline_params is None:
        if "static" in analysis_mode.lower():
            baseline_params = ["D0", "D_offset"]
        else:
            baseline_params = ["D0", "D_offset", "phi0"]

    gradient_norms = compute_gradient_norms(parameters, data, config, analysis_mode)

    baseline_grads = [
        gradient_norms[name] for name in baseline_params if name in gradient_norms
    ]
    if not baseline_grads:
        logger.warning(
            f"No baseline parameters found in gradient norms: {baseline_params}"
        )
        baseline_grads = [1.0]

    baseline_grad = np.exp(np.mean(np.log(np.maximum(baseline_grads, 1e-10))))

    x_scale_map = {}
    for name, grad_norm in gradient_norms.items():
        raw_scale = baseline_grad / max(grad_norm, 1e-10) * safety_factor
        clipped_scale = np.clip(raw_scale, min_scale, max_scale)
        x_scale_map[name] = float(clipped_scale)

        ratio = grad_norm / baseline_grad
        if ratio > 10:
            logger.info(
                f"Parameter {name:18s}: gradient {ratio:>8.0f}x baseline "
                f"-> x_scale={clipped_scale:.2e}"
            )
        elif ratio < 0.1:
            logger.info(
                f"Parameter {name:18s}: gradient {ratio:>8.2f}x baseline "
                f"-> x_scale={clipped_scale:.2e}"
            )
        else:
            logger.debug(
                f"Parameter {name:18s}: gradient {ratio:>8.2f}x baseline "
                f"-> x_scale={clipped_scale:.2e}"
            )

    return x_scale_map


__all__ = [
    "compute_gradient_norms",
    "compute_optimal_x_scale",
]
