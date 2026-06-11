"""Sequential per-angle optimization strategy.

Provides a fallback optimization strategy for when angle-stratified chunking
cannot be used. Each phi angle is optimized independently and the per-angle
results are combined (by default via inverse-variance weighting).

Use cases
---------
- Extreme angle imbalance (ratio > 5.0).
- Stratification explicitly disabled.
- Debugging and validation.
- Memory-constrained environments.

Notes
-----
Author: Homodyne Development Team. Date: 2026-01-14.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from nlsq import LeastSquares

from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)

# Global NLSQ LeastSquares instance for reuse
_nlsq_engine: LeastSquares | None = None


def _get_nlsq_engine() -> LeastSquares:
    """Get or create global NLSQ LeastSquares engine."""
    global _nlsq_engine
    if _nlsq_engine is None:
        _nlsq_engine = LeastSquares(enable_stability=True)
    return _nlsq_engine


def _jax_jacobian(
    func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
) -> np.ndarray:
    """Compute Jacobian using JAX autodiff.

    Uses forward-mode autodiff (jacfwd) which is O(n_params × cost_f),
    efficient when n_params << n_residuals (typical in least squares).

    Parameters
    ----------
    func : callable
        Function to differentiate: f(params) -> residuals
    params : np.ndarray
        Point at which to evaluate Jacobian

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape (n_residuals, n_params)
    """
    # Convert to JAX array
    params_jax = jnp.asarray(params)

    # Wrap function for JAX compatibility
    def jax_func(p: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(func(np.asarray(p)))

    # Compute Jacobian using forward-mode autodiff
    jac_fn = jax.jacfwd(jax_func)
    jac = jac_fn(params_jax)

    return np.asarray(jac)


def _coerce_mapping_to_array(
    mapping: Mapping[Any, Any],
    n_params: int,
    parameter_names: Sequence[str] | None,
    label: str,
) -> np.ndarray:
    """Convert a parameter mapping to a float64 array aligned with solver order."""
    if parameter_names and len(parameter_names) == n_params:
        index_map = {name: idx for idx, name in enumerate(parameter_names)}
        array = np.ones(n_params, dtype=np.float64)
        for key, value in mapping.items():
            if key not in index_map:
                logger.warning(
                    "%s mapping key '%s' not found in parameter_names; ignoring",
                    label,
                    key,
                )
                continue
            array[index_map[key]] = float(value)
        return array

    # Fallback: assume integer indices
    array = np.ones(n_params, dtype=np.float64)
    for key, value in mapping.items():
        try:
            idx = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot align {label} mapping without parameter names; invalid key: {key}"
            ) from exc
        if idx < 0 or idx >= n_params:
            raise ValueError(f"{label} mapping index {idx} out of range for {n_params} parameters")
        array[idx] = float(value)
    return array


def _coerce_numeric_array(
    value: Any,
    n_params: int,
    label: str,
) -> np.ndarray:
    """Ensure a numeric array of length n_params."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(n_params, float(arr), dtype=np.float64)
    elif arr.size != n_params:
        raise ValueError(f"{label} must have {n_params} entries (got shape {arr.shape})")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain finite float64 values")
    return arr.reshape(n_params)


def _normalize_least_squares_kwargs(
    optimizer_kwargs: dict[str, Any],
    n_params: int,
    parameter_names: Sequence[str] | None,
) -> dict[str, Any]:
    """Normalize NLSQ least_squares kwargs to numeric-friendly forms."""
    if not optimizer_kwargs:
        return {}

    normalized: dict[str, Any] = {}

    for key, value in optimizer_kwargs.items():
        normalized[key] = value

    if "x_scale" in normalized:
        x_scale_value = normalized["x_scale"]
        try:
            if isinstance(x_scale_value, Mapping):
                normalized["x_scale"] = _coerce_mapping_to_array(
                    x_scale_value, n_params, parameter_names, "x_scale"
                )
            else:
                normalized["x_scale"] = _coerce_numeric_array(x_scale_value, n_params, "x_scale")
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Dropping non-numeric x_scale due to %s; reverting to default",
                exc,
            )
            normalized.pop("x_scale", None)

    for scalar_key in ("diff_step", "f_scale"):
        if scalar_key not in normalized:
            continue
        raw_value = normalized[scalar_key]
        try:
            if isinstance(raw_value, Mapping):
                normalized[scalar_key] = _coerce_mapping_to_array(
                    raw_value, n_params, parameter_names, scalar_key
                )
            else:
                arr = np.asarray(raw_value, dtype=np.float64)
                if arr.ndim == 0:
                    normalized[scalar_key] = float(arr)
                elif arr.size == n_params:
                    normalized[scalar_key] = _coerce_numeric_array(arr, n_params, scalar_key)
                else:
                    raise ValueError(
                        f"{scalar_key} must be scalar or length {n_params}, got {arr.shape}"
                    )
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Dropping non-numeric %s due to %s; reverting to default",
                scalar_key,
                exc,
            )
            normalized.pop(scalar_key, None)

    for tol_key in ("ftol", "xtol", "gtol"):
        if tol_key not in normalized:
            continue
        try:
            normalized[tol_key] = float(normalized[tol_key])
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Dropping non-numeric %s due to %s; reverting to default",
                tol_key,
                exc,
            )
            normalized.pop(tol_key, None)

    if "max_nfev" in normalized:
        try:
            normalized["max_nfev"] = int(normalized["max_nfev"])
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Dropping non-numeric max_nfev due to %s; reverting to default",
                exc,
            )
            normalized.pop("max_nfev", None)

    if normalized:
        summary = []
        for key, value in normalized.items():
            if isinstance(value, np.ndarray):
                summary.append(f"{key}=array(shape={value.shape}, dtype={value.dtype})")
            else:
                summary.append(f"{key}={type(value).__name__}")
        logger.debug(
            "Sequential least_squares kwargs sanitized: %s",
            "; ".join(summary),
        )

    return normalized


@dataclass
class AngleSubset:
    """Data subset for a single phi angle.

    Attributes
    ----------
    phi_angle : float
        The phi angle value for this subset
    phi_indices : np.ndarray
        Indices where phi == phi_angle
    n_points : int
        Number of data points for this angle
    phi : np.ndarray
        Phi values (all equal to phi_angle)
    t1 : np.ndarray
        Time 1 values
    t2 : np.ndarray
        Time 2 values
    g2_exp : np.ndarray
        Experimental g2 values
    """

    phi_angle: float
    phi_indices: np.ndarray
    n_points: int
    phi: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    g2_exp: np.ndarray


@dataclass
class SequentialResult:
    """Result from sequential per-angle optimization.

    Attributes
    ----------
    combined_parameters : np.ndarray
        Combined optimized parameters (weighted average)
    combined_covariance : np.ndarray
        Combined covariance matrix
    per_angle_results : list[dict]
        Individual results for each angle
    n_angles_optimized : int
        Number of angles successfully optimized
    n_angles_failed : int
        Number of angles that failed optimization
    total_cost : float
        Combined optimization cost
    success_rate : float
        Fraction of angles that converged (0.0-1.0)
    """

    combined_parameters: np.ndarray
    combined_covariance: np.ndarray
    per_angle_results: list[dict[str, Any]]
    n_angles_optimized: int
    n_angles_failed: int
    total_cost: float
    success_rate: float
    initial_jacobian_norms: np.ndarray | None = None
    final_jacobian_norms: np.ndarray | None = None


JAC_SAMPLE_SIZE = 4096


def _select_jacobian_sample(subset: AngleSubset, sample_size: int) -> dict[str, Any]:
    """Select a representative subset of rows for Jacobian diagnostics."""
    size = min(sample_size, subset.n_points)
    if size <= 0:
        return {
            "phi": subset.phi,
            "t1": subset.t1,
            "t2": subset.t2,
            "g2": subset.g2_exp,
            "scale": 1.0,
            "indices": slice(None),
        }

    idx: slice | np.ndarray
    if size == subset.n_points:
        idx = slice(None)
        scale = 1.0
    else:
        idx = np.linspace(0, subset.n_points - 1, size).astype(int)
        scale = np.sqrt(subset.n_points / float(size))

    return {
        "phi": subset.phi[idx],
        "t1": subset.t1[idx],
        "t2": subset.t2[idx],
        "g2": subset.g2_exp[idx],
        "scale": scale,
        "indices": idx,
        "size": size,
    }


def _estimate_initial_jacobian_norms(
    residual_func: Callable,
    params: np.ndarray,
    sample: dict[str, Any],
) -> np.ndarray | None:
    """Estimate column norms at the starting point via JAX autodiff."""
    if sample["phi"].size == 0:
        return None

    def sample_residual_vector(p: np.ndarray) -> np.ndarray:
        return residual_func(p, sample["phi"], sample["t1"], sample["t2"], sample["g2"])

    try:
        jac = _jax_jacobian(sample_residual_vector, params)
        norms = np.linalg.norm(jac, axis=0) * sample.get("scale", 1.0)
        return norms
    except (
        ValueError,
        RuntimeError,
        TypeError,
        np.linalg.LinAlgError,
    ) as exc:  # pragma: no cover - diagnostic
        logger.debug(f"Initial Jacobian estimation failed: {exc}")
        return None


def _compute_final_jacobian_norms(
    jacobian: np.ndarray | None,
    sample: dict[str, Any],
    total_rows: int,
) -> np.ndarray | None:
    """Compute column norms from NLSQ's final Jacobian, using subsampling."""
    if jacobian is None:
        return None

    try:
        if isinstance(sample["indices"], slice):
            jac_subset = jacobian
            scale = 1.0
        else:
            idx = sample["indices"]
            jac_subset = jacobian[idx]
            scale = np.sqrt(total_rows / float(len(idx)))

        norms = np.linalg.norm(jac_subset, axis=0) * scale
        return norms
    except (
        ValueError,
        RuntimeError,
        TypeError,
        IndexError,
        np.linalg.LinAlgError,
    ) as exc:  # pragma: no cover
        logger.debug(f"Final Jacobian norm computation failed: {exc}")
        return None


def split_data_by_angle(
    phi: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    g2_exp: np.ndarray,
    min_points_per_angle: int = 10,
) -> list[AngleSubset]:
    """Split dataset into per-angle subsets.

    Parameters
    ----------
    phi : np.ndarray
        Phi angle values (flattened)
    t1 : np.ndarray
        Time 1 values (flattened)
    t2 : np.ndarray
        Time 2 values (flattened)
    g2_exp : np.ndarray
        Experimental g2 values (flattened)
    min_points_per_angle : int, optional
        Minimum points required per angle, default: 10

    Returns
    -------
    list[AngleSubset]
        List of angle subsets, one per unique phi value

    Raises
    ------
    ValueError
        If any angle has fewer than min_points_per_angle points

    Examples
    --------
    >>> phi = np.array([0, 0, 90, 90, 180, 180])
    >>> t1 = np.linspace(0, 1, 6)
    >>> t2 = np.linspace(0, 1, 6)
    >>> g2 = np.ones(6)
    >>> subsets = split_data_by_angle(phi, t1, t2, g2)
    >>> len(subsets)
    3
    >>> subsets[0].phi_angle
    0.0
    >>> subsets[0].n_points
    2
    """
    # Convert to numpy for indexing
    phi_np = np.asarray(phi)
    t1_np = np.asarray(t1)
    t2_np = np.asarray(t2)
    g2_np = np.asarray(g2_exp)

    # Get unique angles
    unique_angles = np.unique(phi_np)
    logger.info(f"Splitting data into {len(unique_angles)} angle subsets")

    subsets = []
    for angle in unique_angles:
        # Find indices for this angle
        indices = np.where(np.isclose(phi_np, angle, atol=1e-6))[0]
        n_points = len(indices)

        if n_points < min_points_per_angle:
            raise ValueError(
                f"Angle {angle:.2f} deg has only {n_points} points, "
                f"minimum required: {min_points_per_angle}"
            )

        # Extract subset
        subset = AngleSubset(
            phi_angle=float(angle),
            phi_indices=indices,
            n_points=n_points,
            phi=phi_np[indices],
            t1=t1_np[indices],
            t2=t2_np[indices],
            g2_exp=g2_np[indices],
        )

        subsets.append(subset)
        logger.debug(
            f"  Angle {angle:6.2f} deg: {n_points:,} points "
            f"({n_points / len(phi_np) * 100:.1f}% of total)"
        )

    return subsets


def optimize_single_angle(
    subset: AngleSubset,
    residual_func: Callable,
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    **optimizer_kwargs,
) -> dict[str, Any]:
    """Optimize parameters for a single phi angle.

    Parameters
    ----------
    subset : AngleSubset
        Data for this angle
    residual_func : callable
        Residual function: residual_func(params, phi, t1, t2) -> residuals
    initial_params : np.ndarray
        Initial parameter guess
    bounds : tuple of np.ndarray
        (lower_bounds, upper_bounds) for parameters
    **optimizer_kwargs
        Additional arguments passed to NLSQ optimizer

    Returns
    -------
    dict
        Result dictionary with keys:
        - 'parameters': Optimized parameters
        - 'covariance': Covariance matrix
        - 'cost': Final cost
        - 'success': Whether optimization converged
        - 'n_iterations': Number of iterations
        - 'message': Status message
        - 'n_points': Number of points used
        - 'phi_angle': Angle value

    Notes
    -----
    Uses NLSQ LeastSquares for JAX-accelerated optimization.
    """
    logger.debug(f"Optimizing angle {subset.phi_angle:.2f} deg ({subset.n_points:,} points)")

    try:
        # Sanitize dtypes: enforce float64 arrays
        initial_params = np.asarray(initial_params, dtype=np.float64)
        lower_bounds, upper_bounds = bounds
        lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
        upper_bounds = np.asarray(upper_bounds, dtype=np.float64)

        if (
            initial_params.shape[0] != lower_bounds.shape[0]
            or initial_params.shape[0] != upper_bounds.shape[0]
        ):
            raise ValueError(
                "Initial parameters and bounds must have matching shapes: "
                f"init={initial_params.shape}, lower={lower_bounds.shape}, upper={upper_bounds.shape}"
            )

        if not (
            np.all(np.isfinite(initial_params))
            and np.all(np.isfinite(lower_bounds))
            and np.all(np.isfinite(upper_bounds))
        ):
            raise ValueError("Initial parameters and bounds must be finite float64 values")

        # TRF method requires non-degenerate intervals (lower < upper) for all parameters.
        # Fixed parameters (lower == upper) must be handled by the caller before passing to sequential.
        if not np.all(lower_bounds < upper_bounds):
            raise ValueError(
                "Sequential optimizer requires strict lower < upper bounds for all parameters. "
                "Fixed parameters (lower == upper) should be removed before sequential optimization."
            )

        logger.debug(
            "Angle %.2f deg dtype check: init=%s%s lower=%s%s upper=%s%s",
            subset.phi_angle,
            initial_params.dtype,
            initial_params.shape,
            lower_bounds.dtype,
            lower_bounds.shape,
            upper_bounds.dtype,
            upper_bounds.shape,
        )

        # Prepare Jacobian sampling subset for diagnostics
        jac_sample = _select_jacobian_sample(subset, JAC_SAMPLE_SIZE)
        initial_jacobian_norms = _estimate_initial_jacobian_norms(
            residual_func, initial_params, jac_sample
        )

        # Define residual function for this angle (JAX-compatible)
        # Returns jax.Array (jnp.ndarray) because the NLSQ engine JIT-traces
        # this closure; ``np.asarray()`` on the trace would raise
        # TracerArrayConversionError. The return type is intentionally
        # the JAX flavor, not numpy.
        def residuals(params: np.ndarray) -> jnp.ndarray:
            return jnp.asarray(
                residual_func(params, subset.phi, subset.t1, subset.t2, subset.g2_exp)
            )

        # Get NLSQ engine and run optimization
        engine = _get_nlsq_engine()
        result = engine.least_squares(
            fun=residuals,
            x0=initial_params,
            bounds=(lower_bounds, upper_bounds),
            method="trf",
            **optimizer_kwargs,
        )

        # Extract Jacobian for diagnostics (NLSQ returns 'jac' key)
        jac = result.get("jac")
        final_jacobian_norms = _compute_final_jacobian_norms(jac, jac_sample, subset.n_points)

        # Compute covariance if possible
        try:
            # Covariance from Jacobian: (J^T J)^{-1} * s^2
            # where s^2 = sum(r^2) / (n - p) is the residual variance estimate
            if jac is not None:
                # result["fun"] is scalar cost, not residual vector — always recompute
                residuals_final = residuals(result["x"])
                n_residuals = len(residuals_final)
                n_params = len(initial_params)
                s2 = (residuals_final @ residuals_final) / max(n_residuals - n_params, 1)
                try:
                    cov = np.linalg.inv(jac.T @ jac) * s2
                except np.linalg.LinAlgError:
                    try:
                        cov = np.linalg.pinv(jac.T @ jac) * s2
                        logger.warning("Singular J^T J - used pinv fallback for covariance")
                    except np.linalg.LinAlgError:
                        cov = np.eye(len(initial_params))
            else:
                raise ValueError("No Jacobian available")
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to identity if singular
            logger.warning(f"Could not compute covariance for angle {subset.phi_angle:.2f} deg")
            cov = np.eye(len(initial_params))

        # NLSQ result has different key names
        success = result.get("success", False)
        cost = result.get("cost", np.inf)
        n_iterations = result.get("nfev", 0)
        message = result.get("message", "")

        return {
            "parameters": np.asarray(result["x"]),
            "covariance": cov,
            "cost": float(cost),
            "success": bool(success),
            "n_iterations": int(n_iterations),
            "message": str(message),
            "n_points": subset.n_points,
            "phi_angle": subset.phi_angle,
            "jac_initial_norms": initial_jacobian_norms,
            "jac_final_norms": final_jacobian_norms,
        }

    except (ValueError, RuntimeError, OSError) as e:
        logger.error(f"Optimization failed for angle {subset.phi_angle:.2f} deg: {e}")
        return {
            "parameters": initial_params,
            "covariance": np.eye(len(initial_params)),
            "cost": np.inf,
            "success": False,
            "n_iterations": 0,
            "message": f"Failed: {str(e)}",
            "n_points": subset.n_points,
            "phi_angle": subset.phi_angle,
            "jac_initial_norms": None,
            "jac_final_norms": None,
        }


def combine_angle_results(
    per_angle_results: list[dict[str, Any]],
    weighting: str = "inverse_variance",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Combine per-angle optimization results.

    Parameters
    ----------
    per_angle_results : list of dict
        Results from optimize_single_angle for each angle
    weighting : str, optional
        Weighting scheme: 'inverse_variance' | 'uniform' | 'n_points'
        Default: 'inverse_variance' (optimal statistical weighting)

    Returns
    -------
    combined_params : np.ndarray
        Weighted average of parameters
    combined_cov : np.ndarray
        Combined covariance matrix
    total_cost : float
        Sum of individual costs

    Notes
    -----
    Inverse variance weighting:
        w_i = 1 / σ²_i
        μ = Σ(w_i × x_i) / Σ(w_i)
        σ² = 1 / Σ(w_i)

    This provides optimal statistical combination when errors are independent.
    """
    # Filter to successful optimizations
    successful = [r for r in per_angle_results if r["success"]]

    if not successful:
        raise ValueError("No angles converged - cannot combine results")

    logger.info(
        f"Combining results from {len(successful)}/{len(per_angle_results)} "
        f"successful angle optimizations"
    )

    # Extract parameters and covariances
    params_list = np.array([r["parameters"] for r in successful])
    cov_list = np.array([r["covariance"] for r in successful])

    # Compute weights
    if weighting == "inverse_variance":
        # Weight by 1/σ² (diagonal of inverse covariance). Clip the per-angle
        # mean variance to a floor so a singular (near-zero) covariance can't
        # blow the weight up to ~1e10, and reject non-finite variances (NaN/inf
        # from a failed solve) by giving those angles zero weight.
        min_var = 1e-10
        mean_vars = np.array([np.diag(cov).mean() for cov in cov_list])
        finite = np.isfinite(mean_vars)
        weights = np.zeros_like(mean_vars)
        weights[finite] = 1.0 / np.maximum(mean_vars[finite], min_var)
        if not np.any(weights > 0):
            raise ValueError("All angle covariances are non-finite; cannot inverse-variance weight")
    elif weighting == "n_points":
        # Weight by number of data points
        weights = np.array([r["n_points"] for r in successful], dtype=float)
    elif weighting == "uniform":
        # Equal weights
        weights = np.ones(len(successful))
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    # Normalize weights
    # Add small epsilon to prevent division by zero
    weights = weights / (weights.sum() + 1e-10)

    # Weighted average of parameters
    combined_params = np.sum(params_list * weights[:, np.newaxis], axis=0)

    # Combined covariance (inverse variance weighting formula)
    if weighting == "inverse_variance":
        # σ² = 1 / Σ(1/σ²_i)
        # Add small epsilon to prevent division by zero
        inv_vars = np.array([1.0 / (np.diag(cov) + 1e-10) for cov in cov_list])
        # Exclude angles whose per-angle variance was non-finite — those angles
        # were already given zero weight above, so a NaN/inf covariance must not
        # poison the combined inverse-variance covariance either.
        inv_vars = inv_vars[finite]
        combined_var = 1.0 / inv_vars.sum(axis=0)
        combined_cov = np.diag(combined_var)
    else:
        # Weighted average of covariances
        combined_cov = np.sum(cov_list * weights[:, np.newaxis, np.newaxis], axis=0)

    # Total cost
    total_cost = sum(r["cost"] for r in successful)

    logger.info(f"Combined parameters using {weighting} weighting")
    logger.debug(f"  Weights: {weights}")
    logger.debug(f"  Total cost: {total_cost:.4f}")

    return combined_params, combined_cov, total_cost


def strip_fixed_parameters(
    initial_params: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove fixed parameters (lower == upper) from the optimizer inputs.

    The TRF solver used by sequential optimization requires strict
    lower < upper for every parameter.  Fixed parameters (equality
    constraints encoded as lower == upper) must be stripped before the
    call and their known values re-inserted into the result.

    Parameters
    ----------
    initial_params : np.ndarray
        Full parameter vector including fixed parameters.
    lower_bounds : np.ndarray
        Lower bounds array (same length as initial_params).
    upper_bounds : np.ndarray
        Upper bounds array (same length as initial_params).

    Returns
    -------
    free_params : np.ndarray
        Subset of initial_params where lower < upper.
    free_lower : np.ndarray
        Lower bounds for free parameters.
    free_upper : np.ndarray
        Upper bounds for free parameters.
    free_mask : np.ndarray
        Boolean mask (length == len(initial_params)), True where free.

    Examples
    --------
    >>> p = np.array([1.0, 2.0, 3.0])
    >>> lo = np.array([0.0, 2.0, 0.0])
    >>> hi = np.array([5.0, 2.0, 5.0])
    >>> free, fl, fu, mask = strip_fixed_parameters(p, lo, hi)
    >>> free       # array([1.0, 3.0])
    >>> mask       # array([True, False, True])
    """
    free_mask = lower_bounds < upper_bounds
    return (
        initial_params[free_mask],
        lower_bounds[free_mask],
        upper_bounds[free_mask],
        free_mask,
    )


def restore_fixed_parameters(
    free_result: np.ndarray,
    fixed_values: np.ndarray,
    free_mask: np.ndarray,
) -> np.ndarray:
    """Re-insert fixed parameter values into the optimized result.

    Inverse of :func:`strip_fixed_parameters`.

    Parameters
    ----------
    free_result : np.ndarray
        Optimized values for the free parameters.
    fixed_values : np.ndarray
        Full reference parameter vector (fixed positions taken from here).
    free_mask : np.ndarray
        Boolean mask returned by :func:`strip_fixed_parameters`.

    Returns
    -------
    np.ndarray
        Full parameter vector with fixed values restored.
    """
    result = np.array(fixed_values, dtype=np.float64)
    result[free_mask] = free_result
    return result


def optimize_per_angle_sequential(
    phi: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    g2_exp: np.ndarray,
    residual_func: Callable[..., Any],
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    weighting: str = "inverse_variance",
    min_success_rate: float = 0.5,
    parameter_names: Sequence[str] | None = None,
    **optimizer_kwargs,
) -> SequentialResult:
    """Optimize parameters sequentially for each phi angle.

    Main entry point for sequential per-angle optimization.

    Parameters
    ----------
    phi : np.ndarray
        Phi angle values (flattened)
    t1 : np.ndarray
        Time 1 values (flattened)
    t2 : np.ndarray
        Time 2 values (flattened)
    g2_exp : np.ndarray
        Experimental g2 values (flattened)
    residual_func : callable
        Residual function: residual_func(params, phi, t1, t2, g2) -> residuals
    initial_params : np.ndarray
        Initial parameter guess
    bounds : tuple of np.ndarray
        (lower_bounds, upper_bounds)
    weighting : str, optional
        Result combination weighting: 'inverse_variance' | 'uniform' | 'n_points'
    min_success_rate : float, optional
        Minimum fraction of angles that must converge (0.0-1.0), default: 0.5
    parameter_names : Sequence[str], optional
        Parameter ordering used to align per-parameter kwargs (e.g., x_scale)
    **optimizer_kwargs
        Additional arguments passed to NLSQ LeastSquares.least_squares

    Returns
    -------
    SequentialResult
        Combined optimization results

    Raises
    ------
    RuntimeError
        If success rate < min_success_rate

    Examples
    --------
    >>> # Simple example with 3 angles
    >>> phi = np.array([0]*100 + [90]*100 + [180]*100)
    >>> t1 = np.tile(np.linspace(0, 1, 100), 3)
    >>> t2 = np.tile(np.linspace(0, 1, 100), 3)
    >>> g2 = np.ones(300)
    >>>
    >>> def residuals(params, phi, t1, t2, g2):
    ...     # Simple model
    ...     return g2 - (1.0 + params[0] * np.exp(-params[1] * t1))
    >>>
    >>> result = optimize_per_angle_sequential(
    ...     phi, t1, t2, g2,
    ...     residuals,
    ...     initial_params=np.array([0.5, 1.0]),
    ...     bounds=(np.array([0.0, 0.0]), np.array([1.0, 10.0]))
    ... )
    >>> result.success_rate
    1.0
    >>> len(result.per_angle_results)
    3
    """
    logger.info(
        f"Starting sequential per-angle optimization\n"
        f"  Total points: {len(phi):,}\n"
        f"  Unique angles: {len(np.unique(phi))}\n"
        f"  Parameters: {len(initial_params)}\n"
        f"  Weighting: {weighting}"
    )

    optimizer_kwargs = _normalize_least_squares_kwargs(
        optimizer_kwargs,
        n_params=len(initial_params),
        parameter_names=parameter_names,
    )

    # Split data by angle
    subsets = split_data_by_angle(phi, t1, t2, g2_exp)

    # Strip fixed parameters (lower == upper) before passing to TRF solver
    lower_bounds, upper_bounds = bounds
    free_params, free_lower, free_upper, free_mask = strip_fixed_parameters(
        initial_params, lower_bounds, upper_bounds
    )
    free_bounds = (free_lower, free_upper)

    # Re-normalize optimizer kwargs for the reduced (free) parameter count
    has_fixed = np.any(~free_mask)
    if has_fixed:
        optimizer_kwargs = _normalize_least_squares_kwargs(
            optimizer_kwargs,
            n_params=len(free_params),
            parameter_names=(
                [n for n, m in zip(parameter_names, free_mask, strict=False) if m]
                if parameter_names is not None
                else None
            ),
        )
        # Wrap residual_func to expand free params back to full params
        # before evaluation, since residual_func uses hard-coded slicing
        # that expects the full parameter vector
        _original_residual = residual_func

        def residual_func(params, *args, **kwargs):
            full_params = restore_fixed_parameters(params, initial_params, free_mask)
            return _original_residual(full_params, *args, **kwargs)

    # Optimize each angle
    per_angle_results = []
    for subset in subsets:
        result = optimize_single_angle(
            subset,
            residual_func,
            free_params if has_fixed else initial_params,
            free_bounds if has_fixed else bounds,
            **optimizer_kwargs,
        )
        # Restore fixed parameters into the result
        if has_fixed:
            result["parameters"] = restore_fixed_parameters(
                result["parameters"], initial_params, free_mask
            )
        per_angle_results.append(result)

        status = "OK" if result["success"] else "FAIL"
        logger.info(
            f"  {status} Angle {result['phi_angle']:6.2f} deg: "
            f"cost={result['cost']:.4f}, "
            f"iterations={result['n_iterations']}"
        )

    # Aggregate Jacobian diagnostics
    def _aggregate_norms(key: str) -> np.ndarray | None:
        values = [r[key] for r in per_angle_results if r.get(key) is not None]
        if not values:
            return None
        stacked = np.vstack(values)
        return stacked.mean(axis=0)

    aggregated_initial = _aggregate_norms("jac_initial_norms")
    aggregated_final = _aggregate_norms("jac_final_norms")

    # Check success rate
    n_success = sum(1 for r in per_angle_results if r["success"])
    n_total = len(per_angle_results)
    success_rate = n_success / n_total

    if success_rate < min_success_rate:
        raise RuntimeError(
            f"Insufficient convergence: {n_success}/{n_total} angles converged "
            f"({success_rate:.1%}), minimum required: {min_success_rate:.1%}"
        )

    # Combine results
    combined_params, combined_cov, total_cost = combine_angle_results(
        per_angle_results, weighting=weighting
    )

    logger.info(
        f"Sequential optimization complete:\n"
        f"  Success rate: {success_rate:.1%} ({n_success}/{n_total})\n"
        f"  Combined cost: {total_cost:.4f}\n"
        f"  Combined parameters: {combined_params}"
    )

    return SequentialResult(
        combined_parameters=combined_params,
        combined_covariance=combined_cov,
        per_angle_results=per_angle_results,
        n_angles_optimized=n_success,
        n_angles_failed=n_total - n_success,
        total_cost=total_cost,
        success_rate=success_rate,
        initial_jacobian_norms=aggregated_initial,
        final_jacobian_norms=aggregated_final,
    )
