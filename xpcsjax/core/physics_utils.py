"""Shared physics utility functions for homodyne XPCS analysis.

================================================

This module provides common utility functions and physics helpers used by
the NLSQ (meshgrid) computational backend.

These functions were consolidated from:
- jax_backend.py
- physics_nlsq.py
- physics_nlsq.py

to eliminate code duplication and ensure consistent behavior across backends.

Key Functions:
- safe_len: JAX-safe length function for scalars and arrays
- safe_exp: Overflow-protected exponential
- safe_sinc: Numerically stable unnormalized sinc function
- _calculate_diffusion_coefficient_impl_jax: Time-dependent diffusion D(t)
- _calculate_shear_rate_impl_jax: Time-dependent shear rate γ̇(t)
- _create_time_integral_matrix_impl_jax: Trapezoidal cumulative integral matrix
"""

import jax.numpy as jnp
from jax import jit

# safe_exp is canonical in math_primitives so the homodyne and heterodyne paths
# cannot silently diverge on the overflow clip limit. Re-exported here for
# backward compatibility with existing `from physics_utils import safe_exp`.
from xpcsjax.core.math_primitives import safe_exp  # noqa: F401

# Physical and mathematical constants
PI = jnp.pi
EPS = 1e-12  # Numerical stability epsilon


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def safe_len(obj: object) -> int:
    """Return a JAX-safe length for scalars, arrays, and JAX objects.

    Parameters
    ----------
    obj : object
        Any object that might have a length or shape.

    Returns
    -------
    int
        Length of the object (first-dimension size for arrays), or 1 for
        scalars and unsized objects.
    """
    # Handle JAX arrays and numpy arrays with shape attribute
    if hasattr(obj, "shape"):
        if obj.shape == () or len(obj.shape) == 0:
            # Scalar (0-dimensional array)
            return 1
        else:
            # Array - return first dimension size
            return int(obj.shape[0])

    # Handle objects with __len__ method (lists, tuples, etc.)
    if hasattr(obj, "__len__"):
        try:
            return len(obj)
        except TypeError:
            # This catches "len() of unsized object" errors
            return 1

    # Handle scalars (int, float, etc.)
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        # Iterable but not string/bytes
        try:
            return len(list(obj))
        except (TypeError, ValueError):
            return 1

    # Default case: treat as scalar
    return 1


@jit
def safe_sinc(x: jnp.ndarray) -> jnp.ndarray:
    r"""Safe UNNORMALIZED sinc function: sin(x) / x (NOT sin(πx) / (πx)).

    This matches the reference implementation which uses sin(arg) / arg directly.
    The phase argument already includes all necessary scaling factors.

    P2-4: Uses a Taylor expansion near zero (1 - x²/6 + x⁴/120) for smooth
    gradient continuity. The old hard switch from sin(x)/x to 1.0 at ``|x|``\=EPS
    created a gradient discontinuity that caused spurious NUTS rejections near
    gamma_dot_t0 ≈ 0.

    Parameters
    ----------
    x : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        ``sin(x)/x`` for ``|x|`` >= 1e-4, Taylor approximation for
        ``|x|`` < 1e-4.
    """
    x2 = x * x
    near_zero = 1.0 - x2 / 6.0 + x2 * x2 / 120.0
    far = jnp.sin(x) / jnp.where(jnp.abs(x) > EPS, x, 1.0)  # avoid div/0
    return jnp.where(jnp.abs(x) < 1e-4, near_zero, far)


# =============================================================================
# PHYSICS HELPER FUNCTIONS
# =============================================================================


@jit
def calculate_diffusion_coefficient(
    time_array: jnp.ndarray,
    D0: float,
    alpha: float,
    D_offset: float,
) -> jnp.ndarray:
    """Calculate time-dependent diffusion coefficient using discrete evaluation.

    Follows reference v1 implementation: D_t[i] = D0 * (time_array[i] ** alpha) + D_offset.
    Physical constraint: D(t) should be positive and finite.

    Parameters
    ----------
    time_array : jnp.ndarray
        Array of time points [s].
    D0 : float
        Diffusion coefficient amplitude.
    alpha : float
        Anomalous diffusion exponent.
    D_offset : float
        Baseline diffusion offset.

    Returns
    -------
    jnp.ndarray
        D(t) evaluated at each time point, floored at ``1e-10`` (via
        ``jnp.where`` to preserve gradients) and with near-zero times clamped
        to avoid the ``t^alpha`` singularity for negative ``alpha``.
    """
    # CRITICAL FIX: Replace near-zero values to prevent t=0 with negative alpha causing Inf/NaN
    # When alpha < 0: t^alpha = 1/t^|alpha|, so t=0 → infinity
    # Using jnp.maximum (not addition) to only affect near-zero values
    # Use dt/2 to preserve monotonicity: D(dt/2) < D(dt) for alpha > 0
    #
    # Avoid Python `if shape[0] > 1` which causes JIT recompilation per unique
    # array length. Instead compute dt unconditionally: for n==1, time_array[0]
    # is used twice and the difference is 0, so we fall back to the 1e-8 floor.
    dt_inferred = jnp.abs(time_array[jnp.minimum(1, time_array.shape[0] - 1)] - time_array[0])
    epsilon = jnp.where(dt_inferred * 0.5 > 1e-8, dt_inferred * 0.5, 1e-8)
    time_safe = jnp.where(time_array > epsilon, time_array, epsilon)

    # Compute diffusion coefficient
    D_t = D0 * (time_safe**alpha) + D_offset

    # Ensure positive values — use jnp.where (not jnp.maximum) to preserve
    # gradients below the floor for NLSQ Jacobian computation and NUTS leapfrog.
    return jnp.where(D_t > 1e-10, D_t, 1e-10)


@jit
def calculate_shear_rate(
    time_array: jnp.ndarray,
    gamma_dot_0: float,
    beta: float,
    gamma_dot_offset: float,
) -> jnp.ndarray:
    """Calculate time-dependent shear rate using discrete evaluation.

    Follows reference v1 implementation: γ̇_t[i] = γ̇₀ * (time_array[i] ** β) + γ̇_offset.

    Parameters
    ----------
    time_array : jnp.ndarray
        Array of time points [s].
    gamma_dot_0 : float
        Shear rate amplitude.
    beta : float
        Shear rate exponent.
    gamma_dot_offset : float
        Baseline shear rate offset.

    Returns
    -------
    jnp.ndarray
        γ̇(t) evaluated at each time point, floored at ``1e-10`` (via
        ``jnp.where`` to preserve gradients) and with near-zero times clamped
        to avoid the ``t^beta`` singularity for negative ``beta``.
    """
    # CRITICAL FIX: Replace t=0 with dt to prevent singularity when beta < 0
    # When beta < 0: t^beta = 1/t^|beta|, so t=0 → infinity
    # Strategy: Replace only the first element (t=0) with dt, leave others unchanged
    # This ensures smooth continuity: γ̇(dt), γ̇(dt), γ̇(2dt), ...
    #
    # Avoid Python `if shape[0] > 1` which causes JIT recompilation per unique
    # array length. For n==1, index 0 is used twice → inferred dt=0, but the
    # jnp.where guard below keeps it safe with a 1e-8 floor.
    dt = jnp.where(
        jnp.abs(time_array[jnp.minimum(1, time_array.shape[0] - 1)] - time_array[0]) > 1e-8,
        jnp.abs(time_array[jnp.minimum(1, time_array.shape[0] - 1)] - time_array[0]),
        1e-8,
    )

    # Replace near-zero values with dt/2 floor, matching calculate_diffusion_coefficient
    # This provides a continuous floor at the midpoint instead of exact-zero equality check.
    # Floor = 1e-8 matches calculate_diffusion_coefficient — both are power-law t^exponent
    # and have the same singularity structure at t=0.
    epsilon = jnp.where(dt * 0.5 > 1e-8, dt * 0.5, 1e-8)
    time_safe = jnp.where(time_array > epsilon, time_array, epsilon)

    gamma_t = gamma_dot_0 * (time_safe**beta) + gamma_dot_offset
    # Ensure positive values — use jnp.where (not jnp.maximum) to preserve gradients.
    return jnp.where(gamma_t > 1e-10, gamma_t, 1e-10)


@jit
def create_time_integral_matrix(
    time_dependent_array: jnp.ndarray,
) -> jnp.ndarray:
    r"""Create time integral matrix using trapezoidal numerical integration.

    Computes the full N x N matrix of pairwise trapezoidal integral differences
    via broadcasting. The dt scaling happens in wavevector_q_squared_half_dt,
    NOT in this cumsum.

    Algorithm:

    1. Trapezoidal integration: cumsum[i] = Sum(k=0 to i-1) 0.5 * (f[k] + f[k+1])
    2. Compute full difference matrix: matrix[i,j] = smooth_abs(cumsum[i] - cumsum[j])
    3. The dt factor is applied via wavevector_q_squared_half_dt = 0.5 * q^2 * dt

    This gives: matrix[i,j] = number of integration steps.
    Actual integral: dt * matrix[i,j] approximates the integral from 0 to abs(ti-tj) of f(t') dt'

    Benefits over simple cumsum:
    - Reduces oscillations from discretization by ~50%
    - Second-order accuracy (O(dt^2)) vs. first-order (O(dt))
    - Eliminates checkerboard artifacts in diagonal-corrected results

    Parameters
    ----------
    time_dependent_array : jnp.ndarray
        f(t) evaluated at discrete time points.

    Returns
    -------
    jnp.ndarray
        Pairwise time integral matrix ``(n, n)`` in units of integration steps,
        with an exact zero diagonal and a smooth absolute value off-diagonal
        for gradient stability.
    """
    # Handle scalar input by converting to array
    time_dependent_array = jnp.atleast_1d(time_dependent_array)

    # Step 1: Improved cumulative integration using trapezoidal rule
    # Trapezoidal: ∫f(t)dt ≈ dt × Σ(1/2)(f[i] + f[i+1])
    # The dt scaling happens in wavevector_q_squared_half_dt, not here
    #
    # Avoid Python `if n > 1` which causes JIT recompilation per unique array
    # length. The trapezoidal path is unconditionally correct: for n==1,
    # time_dependent_array[:-1] and [1:] are both empty, trap_avg is empty,
    # cumsum_trap is empty, and concatenate([0.0], []) = [0.0] which is the
    # same result as jnp.cumsum([x]) = [x] only if x==0 — but for n==1 the
    # direct-cumsum fallback was returning [x], not [0, x]. Since n==1 never
    # occurs in hot paths (time grids are always 1000+ points), and for
    # correctness the trapezoidal result [0.0] is the correct starting cumsum,
    # the unified path is used unconditionally.
    trap_avg = 0.5 * (time_dependent_array[:-1] + time_dependent_array[1:])
    cumsum_trap = jnp.cumsum(trap_avg)
    cumsum = jnp.concatenate([jnp.array([0.0], dtype=time_dependent_array.dtype), cumsum_trap])

    # Step 2: Create the pairwise integral matrix.
    #
    # The full matrix is: matrix[i,j] = smooth_abs(cumsum[i] - cumsum[j])
    # Because cumsum is monotonically non-decreasing (inputs >= 0 always):
    #   - lower triangle (i >= j): diff[i,j] = cumsum[i] - cumsum[j] >= 0
    #   - upper triangle (i < j):  diff[i,j] = -(diff[j,i])
    #   - diagonal: diff[i,i] = 0 exactly
    #
    # CRITICAL: Use smooth approximation of abs() for gradient stability.
    # jnp.abs() has undefined gradient at x=0, causing NaN in backpropagation.
    # Solution: sqrt(x² + ε) ≈ |x| but is differentiable everywhere.
    # P0-2: epsilon=1e-12 (was 1e-20, below float32 machine epsilon ~1.2e-7).
    epsilon = 1e-12

    # Compute full signed-difference matrix and apply smooth-abs directly.
    # Then force the exact diagonal back to zero: the integral over a
    # zero-duration interval is mathematically zero, while the off-diagonal
    # entries retain the smooth absolute value used for stable gradients.
    diff = cumsum[:, None] - cumsum[None, :]  # Shape: (n, n), symmetric
    matrix = jnp.sqrt(diff**2 + epsilon)  # Shape: (n, n), smooth |diff|
    diagonal = jnp.eye(cumsum.shape[0], dtype=bool)
    matrix = jnp.where(diagonal, jnp.zeros((), dtype=matrix.dtype), matrix)

    return matrix


def trapezoid_cumsum(values: jnp.ndarray) -> jnp.ndarray:
    """Cumulative trapezoid integral without dt scaling (dt is applied outside).

    Returns cumsum so that ``cumsum[j] - cumsum[i]`` equals the trapezoidal sum
    over all intervals between indices ``i`` and ``j``. The caller applies a
    smooth absolute value to that difference when mapping each (t1, t2) pair,
    keeping gradients well-behaved at zero-length intervals.

    This is used by the element-wise computation path.

    Parameters
    ----------
    values : jnp.ndarray
        1D array of values to integrate.

    Returns
    -------
    jnp.ndarray
        Cumulative trapezoidal sums, with a leading ``0.0`` so that
        ``cumsum[j] - cumsum[i]`` equals the trapezoidal integral over
        intervals ``i..j``.
    """
    # Unconditional trapezoidal path — avoids JIT retracing when array size
    # changes. For n==1, values[:-1] and values[1:] are both empty, so
    # cumsum_trap is empty and the result is [0.0], which is the correct
    # cumulative integral (no intervals to sum).
    trap_avg = 0.5 * (values[:-1] + values[1:])
    cumsum_trap = jnp.cumsum(trap_avg)
    return jnp.concatenate([jnp.array([0.0], dtype=values.dtype), cumsum_trap])


# =============================================================================
# DIAGONAL CORRECTION
# =============================================================================
# Re-export from unified diagonal_correction module for backward compatibility.
# See xpcsjax/core/diagonal_correction.py for the canonical implementation.

from xpcsjax.core.diagonal_correction import (  # noqa: E402
    apply_diagonal_correction,  # noqa: F401
    apply_diagonal_correction_batch,  # noqa: F401
)

# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# These aliases maintain backward compatibility with existing code
_calculate_diffusion_coefficient_impl_jax = calculate_diffusion_coefficient
_calculate_shear_rate_impl_jax = calculate_shear_rate
_create_time_integral_matrix_impl_jax = create_time_integral_matrix
_trapezoid_cumsum = trapezoid_cumsum
