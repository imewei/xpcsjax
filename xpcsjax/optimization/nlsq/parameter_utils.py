"""Parameter Utilities for NLSQ Optimization.

Provides utility functions for parameter handling, labeling, status classification,
and per-angle initialization in NLSQ optimization.

Key Functions:
- build_parameter_labels: Create parameter labels with per-angle support
- classify_parameter_status: Identify parameters at bounds
- sample_xdata: Subsample x-data for diagnostic computations
- compute_consistent_per_angle_init: Initialize per-angle params consistently
- compute_jacobian_stats: Compute Jacobian-based statistics
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from xpcsjax.config.types import SCALING_PARAM_NAMES

if TYPE_CHECKING:
    from xpcsjax.config.parameter_manager import ParameterManager

# Physical floor for an accepted per-angle contrast (beta). Real XPCS contrast
# is >= ~1e-3; a fitted value below this is a degenerate/noise-floor result and
# must fall back to the supplied default (see compute_consistent_per_angle_init).
_MIN_PHYSICAL_CONTRAST = 1e-6


@dataclass
class ResolvedPhysicalParameters:
    """Which physical parameters are free vs. fixed for one NLSQ solve.

    ``free_mask`` comes from ``ParameterManager.get_optimizable_parameters()``
    (active-minus-fixed, physics-only by contract). ``values_full`` has the
    CONFIGURED fixed value substituted in at every fixed position -- it is
    NOT simply the caller's original values array.

    When neither ``active_parameters`` nor ``fixed_parameters`` is set (the
    template default), ``free_mask`` is all-``True`` and ``values_full``
    equals the input unchanged -- a provable no-op.
    """

    physical_names: list[str]
    values_full: np.ndarray
    lower_full: np.ndarray
    upper_full: np.ndarray
    free_mask: np.ndarray


def resolve_optimized_physical_parameters(
    param_manager: "ParameterManager",
    physical_names: list[str],
    values_full: np.ndarray,
    lower_full: np.ndarray,
    upper_full: np.ndarray,
    *,
    allow_all_fixed: bool = False,
) -> ResolvedPhysicalParameters:
    """Resolve the free/fixed split AND substitute fixed values for homodyne.

    Physical-parameter-only by design (grilling round 1 Q1) -- a scaling name
    in ``fixed_parameters`` is a hard fit-time error (grilling round 1 Q5, Q8).

    Raises
    ------
    ValueError
        If ``fixed_parameters`` names a scaling parameter, or if the
        resulting free set is empty and ``allow_all_fixed`` is False.
    """
    fixed_params = param_manager.get_fixed_parameters()
    scaling_fixed = [name for name in fixed_params if name in SCALING_PARAM_NAMES]
    if scaling_fixed:
        raise ValueError(
            f"fixed_parameters names scaling parameter(s) {scaling_fixed!r}, "
            "which is not supported for this analysis mode -- fixed_parameters "
            "only constrains physical parameters here. Use 'per_angle_scaling' "
            "initial values to control contrast/offset instead."
        )

    optimizable = set(param_manager.get_optimizable_parameters())
    free_mask = np.array([name in optimizable for name in physical_names], dtype=bool)

    values_full = np.array(values_full, dtype=np.float64)  # copy -- never mutate caller's array
    for i, name in enumerate(physical_names):
        if name in fixed_params:
            values_full[i] = fixed_params[name]

    if not allow_all_fixed and not free_mask.any():
        raise ValueError(
            "fixed_parameters/active_parameters leave nothing left to "
            f"optimize: every physical parameter in {physical_names!r} is "
            "fixed or excluded. Free at least one parameter."
        )

    return ResolvedPhysicalParameters(
        physical_names=list(physical_names),
        values_full=values_full,
        lower_full=np.asarray(lower_full, dtype=np.float64),
        upper_full=np.asarray(upper_full, dtype=np.float64),
        free_mask=free_mask,
    )


def strip_by_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return only the entries of ``values`` where ``mask`` is True."""
    return np.asarray(values)[mask]


def restore_by_mask_numpy(
    free_values: np.ndarray, full_values: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Re-insert solved free values into a full-length array (post-solve only).

    NumPy indexed assignment -- safe only on already-concrete arrays. Do NOT
    call this inside a JAX-traced closure; use :func:`restore_by_mask_jax`.
    """
    result = np.array(full_values, dtype=np.float64)
    result[mask] = np.asarray(free_values)
    return result


def restore_by_mask_jax(free_values, full_values: np.ndarray, mask: np.ndarray):
    """JAX-traceable equivalent of :func:`restore_by_mask_numpy`.

    Uses immutable ``.at[].set()`` -- safe inside a function JAX traces for
    JIT/autodiff (model/residual closures passed to ``curve_fit``/CMA-ES).
    """
    import jax.numpy as jnp

    full_jnp = jnp.asarray(full_values)
    free_idx = jnp.asarray(np.where(mask)[0])
    return full_jnp.at[free_idx].set(jnp.asarray(free_values))


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


def build_parameter_labels(
    per_angle_scaling: bool,
    n_phi: int,
    physical_param_names: list[str],
) -> list[str]:
    """Build parameter labels including per-angle scaling parameters.

    Parameters
    ----------
    per_angle_scaling : bool
        Whether per-angle contrast/offset are used
    n_phi : int
        Number of phi angles
    physical_param_names : list[str]
        Names of physical parameters

    Returns
    -------
    list[str]
        Full list of parameter labels
    """
    labels: list[str] = []
    if per_angle_scaling:
        labels.extend([f"contrast[{i}]" for i in range(n_phi)])
        labels.extend([f"offset[{i}]" for i in range(n_phi)])
    else:
        # Scalar scaling: contrast + offset are always fit (c2 = contrast*g1^2 +
        # offset), so the parameter vector includes them even without per-angle
        # scaling. Emitting them keeps labels aligned with the vector, which
        # always begins with the scaling pair (see
        # ParameterManager.get_all_parameter_names).
        labels.extend(["contrast", "offset"])
    labels.extend(physical_param_names)
    return labels


def classify_parameter_status(
    values: np.ndarray,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
    atol: float = 1e-9,
) -> list[str]:
    """Classify parameters as active or at bounds.

    Parameters
    ----------
    values : np.ndarray
        Current parameter values
    lower : np.ndarray | None
        Lower bounds
    upper : np.ndarray | None
        Upper bounds
    atol : float, optional
        Absolute tolerance for bound comparison

    Returns
    -------
    list[str]
        Status for each parameter: 'active', 'at_lower_bound', or 'at_upper_bound'
    """
    if lower is None or upper is None:
        return ["active"] * len(values)

    statuses: list[str] = []
    for value, lo, hi in zip(values, lower, upper, strict=False):
        if np.isclose(value, lo, atol=atol * (1.0 + abs(lo)), rtol=0.0):
            statuses.append("at_lower_bound")
        elif np.isclose(value, hi, atol=atol * (1.0 + abs(hi)), rtol=0.0):
            statuses.append("at_upper_bound")
        else:
            statuses.append("active")
    return statuses


def sample_xdata(xdata: np.ndarray, max_points: int) -> np.ndarray:
    """Subsample x-data for diagnostic computations.

    Parameters
    ----------
    xdata : np.ndarray
        Input data
    max_points : int
        Maximum number of points to return

    Returns
    -------
    np.ndarray
        Subsampled data
    """
    if max_points <= 0 or xdata.size <= max_points:
        return xdata
    indices = np.linspace(0, xdata.size - 1, max_points, dtype=np.int64)
    return xdata[indices]


def compute_jacobian_stats(
    residual_fn: Callable[..., Any],
    x_subset: np.ndarray,
    params: np.ndarray,
    scaling_factor: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute Jacobian statistics for diagnostics.

    Parameters
    ----------
    residual_fn : Callable
        Residual function
    x_subset : np.ndarray
        Subset of x data for computation
    params : np.ndarray
        Current parameters
    scaling_factor : float
        Scaling factor for statistics

    Returns
    -------
    tuple
        (J^T J matrix, column norms) or (None, None) on failure
    """
    try:
        params_jnp = jnp.asarray(params)
        if hasattr(residual_fn, "jax_residual"):

            def residual_vector(p):
                return jnp.asarray(residual_fn.jax_residual(jnp.asarray(p))).reshape(-1)

        else:

            def residual_vector(p):
                return jnp.asarray(residual_fn(x_subset, *tuple(p))).reshape(-1)

        jac = jax.jacfwd(residual_vector)(params_jnp)
        jac_np = np.asarray(jac)

        # For ill-conditioned Jacobians (cond > 1e6), compute J^T J via QR
        # (J^T J = R^T R) instead of the direct product, which squares the
        # condition number and destabilizes the downstream pinv-based
        # covariance solve. Mirrors jacobian.py's compute_jacobian_stats.
        try:
            cond_number = np.linalg.cond(jac_np)
        except np.linalg.LinAlgError:
            cond_number = np.inf

        if cond_number > 1e6:
            _q, r = np.linalg.qr(jac_np)
            jtj = r.T @ r * scaling_factor
        else:
            jtj = jac_np.T @ jac_np * scaling_factor

        col_norms = np.linalg.norm(jac_np, axis=0) * np.sqrt(scaling_factor)
        return jtj, col_norms
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        from xpcsjax.utils.logging import get_logger as _get_logger

        _get_logger(__name__).debug(f"Could not compute Jacobian stats: {e}")
        return None, None


def compute_consistent_per_angle_init(
    stratified_data: Any,
    physical_params: np.ndarray,
    physical_param_names: list[str],
    default_contrast: float = 0.5,
    default_offset: float = 1.0,
    logger: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-angle contrast/offset consistent with initial physical parameters.

    This function solves a critical initialization problem in laminar_flow mode:
    when physical shear parameters (gamma_dot_t0) are nonzero, the model predicts
    DIFFERENT g2 values at different angles. If per-angle contrast/offset are
    initialized uniformly, large initial residuals can cause the optimizer to
    incorrectly reduce gamma_dot_t0 to zero.

    Instead, we compute per-angle contrast/offset by fitting:
        g2_data[angle] ≈ offset[angle] + contrast[angle] × g1_model²[angle]

    where g1_model is computed using the initial physical parameters.

    Parameters
    ----------
    stratified_data : StratifiedData
        Data containing per-angle g2, phi, t1, t2 arrays
    physical_params : np.ndarray
        Initial physical parameters [D0, alpha, D_offset, (gamma_dot_t0, beta, gamma_dot_t_offset, phi0)]
    physical_param_names : list[str]
        Names of physical parameters to determine analysis mode
    default_contrast : float
        Default contrast value if fitting fails
    default_offset : float
        Default offset value if fitting fails
    logger : logging.Logger, optional
        Logger for diagnostic messages

    Returns
    -------
    contrast_per_angle : np.ndarray
        Per-angle contrast values consistent with physical params
    offset_per_angle : np.ndarray
        Per-angle offset values consistent with physical params
    """
    from xpcsjax.core.physics_utils import (
        calculate_diffusion_coefficient,
        calculate_shear_rate,
        safe_sinc,
        trapezoid_cumsum,
    )

    # Extract data by angle
    if hasattr(stratified_data, "chunks"):
        phi_unique = np.array(
            sorted({phi for chunk in stratified_data.chunks for phi in chunk.phi.tolist()})
        )
        n_phi = len(phi_unique)
        # Get metadata from first chunk
        first_chunk = stratified_data.chunks[0]
        q = first_chunk.q
        L = first_chunk.L
        dt = first_chunk.dt
        t1_unique = np.array(
            sorted({t for chunk in stratified_data.chunks for t in chunk.t1.tolist()})
        )
    else:
        phi_unique = np.unique(stratified_data.phi_flat)
        n_phi = len(phi_unique)
        q = stratified_data.q
        L = stratified_data.L
        dt = stratified_data.dt
        t1_unique = np.unique(stratified_data.t1_flat)

    # Extract physical parameters
    is_laminar_flow = "gamma_dot_t0" in physical_param_names
    D0 = physical_params[0]
    alpha = physical_params[1]
    D_offset = physical_params[2]
    if is_laminar_flow:
        gamma_dot_t0 = physical_params[3]
        beta = physical_params[4]
        gamma_dot_t_offset = physical_params[5]
        phi0 = physical_params[6]
    else:
        gamma_dot_t0 = 0.0
        beta = 0.0
        gamma_dot_t_offset = 0.0
        phi0 = 0.0

    # Precompute time-dependent quantities (same for all angles)
    t_values = np.asarray(t1_unique)
    D_t = calculate_diffusion_coefficient(t_values, D0, alpha, D_offset)
    D_cumsum = np.asarray(trapezoid_cumsum(D_t))
    wavevector_q_squared_half_dt = 0.5 * (q**2) * dt

    if is_laminar_flow:
        gamma_t = calculate_shear_rate(t_values, gamma_dot_t0, beta, gamma_dot_t_offset)
        gamma_cumsum = np.asarray(trapezoid_cumsum(gamma_t))
        sinc_prefactor = 0.5 / np.pi * q * L * dt

    # Initialize output arrays
    contrast_per_angle = np.full(n_phi, default_contrast)
    offset_per_angle = np.full(n_phi, default_offset)

    # Process each angle
    for i, phi in enumerate(phi_unique):
        try:
            # Get data for this angle
            if hasattr(stratified_data, "chunks"):
                # Find data in chunks
                g2_list = []
                t1_list = []
                t2_list = []
                for chunk in stratified_data.chunks:
                    mask = np.isclose(chunk.phi, phi, atol=0.1)
                    if np.any(mask):
                        g2_list.extend(chunk.g2[mask].tolist())
                        t1_list.extend(chunk.t1[mask].tolist())
                        t2_list.extend(chunk.t2[mask].tolist())
                if not g2_list:
                    continue
                g2_data = np.array(g2_list)
                t1_data = np.array(t1_list)
                t2_data = np.array(t2_list)
            else:
                mask = np.isclose(stratified_data.phi_flat, phi, atol=0.1)
                if not np.any(mask):
                    continue
                g2_data = stratified_data.g2_flat[mask]
                t1_data = stratified_data.t1_flat[mask]
                t2_data = stratified_data.t2_flat[mask]

            # Compute g1_model for each data point at this angle.
            # NOTE: Both t1 and t2 index into t1_unique because XPCS correlation
            # matrices C2(t1, t2) use a shared time grid (t1_unique == t2_unique).
            # The physics model computes D(t) on this single grid, and differences
            # D_cumsum[t1_idx] - D_cumsum[t2_idx] give the integral over |t1-t2|.
            t1_idx = np.clip(np.searchsorted(t1_unique, t1_data), 0, len(t1_unique) - 1)
            t2_idx = np.clip(np.searchsorted(t1_unique, t2_data), 0, len(t1_unique) - 1)

            # Diffusion term
            D_diff = D_cumsum[t1_idx] - D_cumsum[t2_idx]
            D_integral_batch = np.abs(D_diff)
            log_g1_diff = -wavevector_q_squared_half_dt * D_integral_batch
            g1_diffusion = np.exp(np.clip(log_g1_diff, -700.0, 0.0))

            if is_laminar_flow:
                # Shear term
                angle_diff = np.deg2rad(phi0 - phi)
                cos_phi = np.cos(angle_diff)
                gamma_diff = gamma_cumsum[t1_idx] - gamma_cumsum[t2_idx]
                gamma_integral_batch = np.abs(gamma_diff)
                sinc_arg = sinc_prefactor * cos_phi * gamma_integral_batch
                sinc_val = safe_sinc(sinc_arg)
                g1_shear = sinc_val**2
                g1_model = g1_diffusion * g1_shear
            else:
                g1_model = g1_diffusion

            # Clip for numerical stability (g1 ∈ [0, 1] by physics). Heavy
            # clipping means the fit diverged out of the physical range; count
            # it so a heavily-clipped (unreliable) result is not silently
            # indistinguishable from a well-behaved one.
            n_clipped = int(np.count_nonzero((g1_model < 1e-10) | (g1_model > 1.0)))
            if n_clipped:
                from xpcsjax.utils.logging import get_logger as _get_logger

                _get_logger(__name__).warning(
                    "g1 model clipped to [1e-10, 1.0] at %d/%d points; "
                    "contrast/offset estimate may be unreliable.",
                    n_clipped,
                    int(g1_model.size),
                )
            g1_model = np.clip(g1_model, 1e-10, 1.0)
            g1_sq = g1_model**2

            # Linear regression: g2 = offset + contrast × g1²
            if len(g2_data) > 2:
                A = np.column_stack([np.ones_like(g1_sq), g1_sq])
                result = np.linalg.lstsq(A, g2_data, rcond=None)
                fit_offset, fit_contrast = result[0]

                # Sanity checks. The contrast floor is a physical minimum, not
                # ``> 0``: a degenerate (flat-g2) fit yields a contrast at the
                # noise floor whose sign is BLAS-dependent — Linux OpenBLAS
                # returns <= 0 (rejected), macOS Accelerate returns ~1e-11
                # (a tiny positive that a bare ``> 0`` test wrongly accepts).
                # Requiring a real minimum makes the fall-back to defaults
                # platform-agnostic.
                if _MIN_PHYSICAL_CONTRAST < fit_contrast < 2.0 and 0.5 < fit_offset < 1.5:
                    contrast_per_angle[i] = fit_contrast
                    offset_per_angle[i] = fit_offset

        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            if logger:
                logger.debug(f"Failed to compute consistent init for angle {phi}: {e}")
            # Keep default values

    if logger:
        logger.info(
            f"Computed consistent per-angle initialization:\n"
            f"  Contrast range: [{contrast_per_angle.min():.4f}, {contrast_per_angle.max():.4f}]\n"
            f"  Offset range: [{offset_per_angle.min():.4f}, {offset_per_angle.max():.4f}]"
        )

    return contrast_per_angle, offset_per_angle


def compute_quantile_per_angle_scaling(
    stratified_data: Any,
    contrast_bounds: tuple[float, float] = (0.0, 1.0),
    offset_bounds: tuple[float, float] = (0.5, 1.5),
    lag_floor_quantile: float = 0.80,
    lag_ceiling_quantile: float = 0.20,
    value_quantile_low: float = 0.10,
    value_quantile_high: float = 0.90,
    logger: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-angle contrast/offset from quantiles of c2_experimental values.

    This function uses physics-informed quantile analysis to estimate contrast and
    offset for each phi angle independently. Unlike least-squares fitting, this
    approach does not require a model and directly extracts scaling from the data.

    Physics basis:
        C2 = contrast × g1² + offset

        - At large time lags, g1² → 0, so C2 → offset (the "floor")
        - At small time lags, g1² ≈ 1, so C2 ≈ contrast + offset (the "ceiling")

    Parameters
    ----------
    stratified_data : StratifiedData
        Data containing per-angle g2_flat, phi_flat, t1_flat, t2_flat arrays.
    contrast_bounds : tuple[float, float]
        Valid bounds for contrast parameter.
    offset_bounds : tuple[float, float]
        Valid bounds for offset parameter.
    lag_floor_quantile : float
        Quantile threshold for "large lag" region (default: 0.80 = top 20% of lags).
    lag_ceiling_quantile : float
        Quantile threshold for "small lag" region (default: 0.20 = bottom 20% of lags).
    value_quantile_low : float
        Quantile for robust floor estimation (default: 0.10).
    value_quantile_high : float
        Quantile for robust ceiling estimation (default: 0.90).
    logger : logging.Logger, optional
        Logger for diagnostic messages.

    Returns
    -------
    contrast_per_angle : np.ndarray
        Per-angle contrast values from quantile estimation.
    offset_per_angle : np.ndarray
        Per-angle offset values from quantile estimation.

    Notes
    -----
    The estimation is robust to outliers by using quantiles instead of min/max.
    The lag-based segmentation ensures we sample from appropriate regions of
    the correlation decay curve.

    This function is designed for the "constant" mode in anti-degeneracy defense,
    where per-angle contrast/offset are estimated once and treated as fixed
    parameters during optimization.
    """
    from xpcsjax.utils.logging import get_logger as _get_logger

    if logger is None:
        logger = _get_logger(__name__)

    # Extract data from stratified_data
    if hasattr(stratified_data, "chunks"):
        # ChunkedData format
        phi_unique = np.array(
            sorted({phi for chunk in stratified_data.chunks for phi in chunk.phi.tolist()})
        )
        n_phi = len(phi_unique)

        # Collect all data into flat arrays
        g2_list = []
        t1_list = []
        t2_list = []
        phi_list = []
        for chunk in stratified_data.chunks:
            g2_list.extend(chunk.g2.tolist())
            t1_list.extend(chunk.t1.tolist())
            t2_list.extend(chunk.t2.tolist())
            phi_list.extend(chunk.phi.tolist())
        g2_flat = np.array(g2_list)
        t1_flat = np.array(t1_list)
        t2_flat = np.array(t2_list)
        phi_flat = np.array(phi_list)
    elif hasattr(stratified_data, "phi_flat"):
        # StratifiedData format with flat per-point arrays
        phi_unique = np.unique(stratified_data.phi_flat)
        n_phi = len(phi_unique)
        g2_flat = stratified_data.g2_flat
        t1_flat = stratified_data.t1_flat
        t2_flat = stratified_data.t2_flat
        phi_flat = stratified_data.phi_flat
    else:
        # Raw (non-stratified, <100k) grid object: phi/t1/t2 are unique grid axes
        # of DIFFERENT lengths and g2 is a dense (n_phi, n_t1, n_t2) cube. Flatten
        # the meshgrid to per-point flats so the rest of the helper is unchanged.
        # This is the path that previously AttributeError'd on .phi_flat
        # (CLAUDE.md / spec §5: <100k unstratified path).
        phi_grid = np.asarray(stratified_data.phi, dtype=np.float64)
        t1_grid = np.asarray(stratified_data.t1, dtype=np.float64)
        t2_grid = np.asarray(stratified_data.t2, dtype=np.float64)
        g2_cube = np.asarray(stratified_data.g2, dtype=np.float64)
        phi_unique = np.unique(phi_grid)
        n_phi = len(phi_unique)
        # Broadcast the (n_phi, n_t1, n_t2) cube onto per-point flats.
        n_t1 = len(t1_grid)
        n_t2 = len(t2_grid)
        t1_mesh, t2_mesh = np.meshgrid(t1_grid, t2_grid, indexing="ij")
        t1_per_angle = t1_mesh.ravel()
        t2_per_angle = t2_mesh.ravel()
        phi_flat = np.repeat(phi_grid, n_t1 * n_t2)
        t1_flat = np.tile(t1_per_angle, n_phi)
        t2_flat = np.tile(t2_per_angle, n_phi)
        g2_flat = g2_cube.reshape(n_phi, n_t1 * n_t2).ravel()

    # Pre-compute time lags (vectorized)
    delta_t = np.abs(t1_flat - t2_flat)

    # Pre-compute midpoint defaults
    contrast_mid = (contrast_bounds[0] + contrast_bounds[1]) / 2.0
    offset_mid = (offset_bounds[0] + offset_bounds[1]) / 2.0

    # Initialize output arrays with defaults
    contrast_per_angle = np.full(n_phi, contrast_mid)
    offset_per_angle = np.full(n_phi, offset_mid)

    # Pre-compute per-point angle indices for exact matching
    phi_indices = np.searchsorted(phi_unique, phi_flat)
    phi_indices = np.clip(phi_indices, 0, n_phi - 1)

    # Process each angle
    for i, phi in enumerate(phi_unique):
        # Get mask for this angle — use exact index matching (not tolerance-based)
        # to avoid bleeding adjacent phi bins on dense angular grids.
        mask = phi_indices == i
        n_points = np.sum(mask)

        if n_points < 100:
            logger.debug(
                f"Angle {i} (phi={phi:.1f} deg): insufficient data ({n_points} points), "
                f"using midpoint defaults"
            )
            continue

        # Extract data for this angle
        c2_angle = g2_flat[mask]
        delta_t_angle = delta_t[mask]

        # Find lag thresholds for this angle
        lag_threshold_high = np.nanpercentile(delta_t_angle, lag_floor_quantile * 100)
        lag_threshold_low = np.nanpercentile(delta_t_angle, lag_ceiling_quantile * 100)

        # OFFSET estimation: From large-lag region where g1² ≈ 0
        large_lag_mask = delta_t_angle >= lag_threshold_high
        if np.sum(large_lag_mask) >= 10:
            c2_floor_region = c2_angle[large_lag_mask]
            offset_est = np.nanpercentile(c2_floor_region, value_quantile_low * 100)
            if not np.isfinite(offset_est):
                offset_est = offset_mid
        else:
            # Fallback: use overall low quantile
            offset_est = np.nanpercentile(c2_angle, value_quantile_low * 100)
            if not np.isfinite(offset_est):
                offset_est = offset_mid

        # Clip offset to bounds
        offset_est = float(np.clip(offset_est, offset_bounds[0], offset_bounds[1]))

        # CONTRAST estimation: From small-lag region where g1² ≈ 1
        small_lag_mask = delta_t_angle <= lag_threshold_low
        if np.sum(small_lag_mask) >= 10:
            c2_ceiling_region = c2_angle[small_lag_mask]
            c2_ceiling = np.nanpercentile(c2_ceiling_region, value_quantile_high * 100)
            if not np.isfinite(c2_ceiling):
                c2_ceiling = contrast_mid + offset_mid
        else:
            # Fallback: use overall high quantile
            c2_ceiling = np.nanpercentile(c2_angle, value_quantile_high * 100)
            if not np.isfinite(c2_ceiling):
                c2_ceiling = contrast_mid + offset_mid

        # contrast ≈ c2_ceiling - offset
        contrast_est = c2_ceiling - offset_est

        # Clip contrast to bounds
        contrast_est = float(np.clip(contrast_est, contrast_bounds[0], contrast_bounds[1]))

        contrast_per_angle[i] = contrast_est
        offset_per_angle[i] = offset_est

        logger.debug(
            f"Angle {i} (phi={phi:.1f} deg): quantile estimation "
            f"contrast={contrast_est:.4f}, offset={offset_est:.4f} "
            f"from {n_points:,} points"
        )

    logger.info(
        f"Quantile-based per-angle estimation complete:\n"
        f"  Contrast range: [{contrast_per_angle.min():.4f}, {contrast_per_angle.max():.4f}]\n"
        f"  Offset range: [{offset_per_angle.min():.4f}, {offset_per_angle.max():.4f}]"
    )

    return contrast_per_angle, offset_per_angle


__all__ = [
    "ResolvedPhysicalParameters",
    "resolve_optimized_physical_parameters",
    "strip_by_mask",
    "restore_by_mask_numpy",
    "restore_by_mask_jax",
    "strip_fixed_parameters",
    "restore_fixed_parameters",
    "build_parameter_labels",
    "classify_parameter_status",
    "sample_xdata",
    "compute_jacobian_stats",
    "compute_consistent_per_angle_init",
    "compute_quantile_per_angle_scaling",
]
