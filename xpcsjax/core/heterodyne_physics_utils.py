"""Numerically safe mathematical primitives for heterodyne physics.

All functions are designed to be compatible with both NumPy and JAX
arrays, avoiding NaN/Inf from edge cases (division by zero,
overflow in exp, negative bases in power).

Shared utilities used by both the NLSQ meshgrid path and the CMC
element-wise path:
- ``trapezoid_cumsum``: O(dt²) cumulative integral
- ``create_time_integral_matrix``: N×N from cumsum (NLSQ only)
- ``smooth_abs``: gradient-safe ``|x|`` for NUTS
- ``compute_transport_rate``: J(t) = D0·t^α + offset
- ``compute_velocity_rate``: v(t) = v0·t^β + v_offset
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Numerically safe math primitives
# ---------------------------------------------------------------------------


def safe_exp(x: jnp.ndarray | np.ndarray, limit: float = 500.0) -> jnp.ndarray:
    """Exponential with overflow protection.

    Clips the argument to [-limit, limit] before computing exp()
    to avoid Inf outputs. The default limit of 500 gives exp(500) ≈ 1.4e217
    which is within float64 range.

    Args:
        x: Input array
        limit: Clipping threshold (symmetric)

    Returns:
        exp(clip(x)), same shape as x
    """
    x = jnp.asarray(x)
    return jnp.exp(jnp.clip(x, -limit, limit))


def safe_power(base: jnp.ndarray | np.ndarray, exponent: float) -> jnp.ndarray:
    """Power function safe for non-positive bases.

    For base ≤ 0, returns 0.0 (the physical limit for t^α transport).
    For base > 0, returns base^exponent normally.

    Args:
        base: Base array (typically time values)
        exponent: Power exponent

    Returns:
        Safe power result, same shape as base
    """
    base = jnp.asarray(base)
    # Use jnp.where instead of jnp.maximum to preserve gradients below the
    # floor: jnp.maximum zeros the gradient when base < 1e-30, which stalls
    # the NLSQ Jacobian and NUTS leapfrog steps.
    base_safe = jnp.where(base > 1e-30, base, 1e-30)
    result = jnp.power(base_safe, exponent)
    return jnp.where(base > 0, result, 0.0)


def safe_divide(
    numerator: jnp.ndarray | np.ndarray,
    denominator: jnp.ndarray | np.ndarray,
    fill: float = 0.0,
    min_denom: float = 1e-30,
) -> jnp.ndarray:
    """Division with protection against zero/near-zero denominators.

    Args:
        numerator: Dividend array
        denominator: Divisor array
        fill: Value to return where denominator is too small
        min_denom: Minimum absolute denominator value

    Returns:
        Safe quotient, same shape as inputs
    """
    num = jnp.asarray(numerator)
    den = jnp.asarray(denominator)
    # Preserve sign of denominator for the floor value; use jnp.where to
    # avoid sign(0.0)=0 which would produce safe_den=0 and intermediate NaN.
    floor = jnp.where(den >= 0, min_denom, -min_denom)
    safe_den = jnp.where(jnp.abs(den) > min_denom, den, floor)
    # Where original denominator was ~0, return fill value
    result = num / safe_den
    return jnp.where(jnp.abs(den) > min_denom, result, fill)


def safe_log(x: jnp.ndarray | np.ndarray, floor: float = 1e-30) -> jnp.ndarray:
    """Logarithm with protection against non-positive arguments.

    Args:
        x: Input array
        floor: Minimum value before taking log

    Returns:
        log(max(x, floor)), same shape as x
    """
    x = jnp.asarray(x)
    # Use jnp.where to preserve gradients: jnp.maximum zeros the gradient
    # when x < floor, stalling log-space parameter updates.
    return jnp.log(jnp.where(x > floor, x, floor))


def safe_sqrt(x: jnp.ndarray | np.ndarray) -> jnp.ndarray:
    """Square root with protection against negative arguments.

    Args:
        x: Input array

    Returns:
        sqrt(max(x, 0)), same shape as x
    """
    x = jnp.asarray(x)
    # Use jnp.where to preserve gradients: jnp.maximum zeros the gradient
    # when x < 0, which would stall the Jacobian at the sqrt floor.
    return jnp.sqrt(jnp.where(x > 0.0, x, 0.0))


def compute_relative_difference(
    a: jnp.ndarray | np.ndarray,
    b: jnp.ndarray | np.ndarray,
) -> jnp.ndarray:
    """Compute element-wise relative difference ``|a - b|`` / max(``|a|``, ``|b|``, 1e-10).

    Useful for comparing correlation matrices or parameter arrays
    where absolute differences may mislead at different scales.

    Args:
        a: First array
        b: Second array

    Returns:
        Relative difference array, values in [0, 2]
    """
    a, b = jnp.asarray(a), jnp.asarray(b)
    max_abs = jnp.maximum(jnp.maximum(jnp.abs(a), jnp.abs(b)), 1e-10)
    return jnp.abs(a - b) / max_abs


def symmetrize(matrix: jnp.ndarray | np.ndarray) -> jnp.ndarray:
    """Force a matrix to be exactly symmetric: (M + M^T) / 2.

    Args:
        matrix: Square matrix

    Returns:
        Symmetric matrix
    """
    m = jnp.asarray(matrix)
    return 0.5 * (m + m.T)


# ---------------------------------------------------------------------------
# Shared integral and rate primitives (used by both NLSQ and CMC paths)
# ---------------------------------------------------------------------------


def smooth_abs(x: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Gradient-safe absolute value: sqrt(x² + ε).

    ``jnp.abs(x)`` has undefined gradient at x=0, which causes NaN
    in NUTS backpropagation on matrix diagonals where integrals are
    zero.  This smooth approximation matches ``|x|`` to O(√ε) and
    has well-defined gradients everywhere.

    Args:
        x: Input array.
        eps: Smoothing parameter. 1e-12 gives ~1e-6 bias on diagonal.

    Returns:
        Smooth ``|x|``, same shape as x.
    """
    return jnp.sqrt(x**2 + eps)


def smooth_clip(
    x: jnp.ndarray,
    low: float,
    high: float,
    sharpness: float = 50.0,
) -> jnp.ndarray:
    """Soft clip to ``[low, high]`` with continuous gradient at the boundaries.

    Acts as the identity in the interior and softplus-smoothed at the
    boundaries.  Use this for physical bounds (e.g. sample fraction in
    [0, 1]) where a hard ``jnp.clip`` would zero the gradient and stall
    NUTS leapfrog integration or NLSQ Jacobian descent (CLAUDE.md rule #7).

    The boundary smear scales as ``1/sharpness`` — at the default value
    ``sharpness=50`` the boundary lands within ~``(high-low)/50`` of the
    target (≈2% of the range), with monotonic identity in the interior.
    Raise ``sharpness`` for a tighter approximation at the cost of
    gradient magnitude near the boundary; lower it for stronger
    regularisation.

    Args:
        x: Input array (any shape).
        low: Lower physical bound (inclusive in the limit).
        high: Upper physical bound (inclusive in the limit).
        sharpness: Softplus sharpness; default 50 gives ~2% boundary smear.

    Returns:
        Smoothly bounded array, asymptotically in (low, high), with
        well-defined gradients everywhere.
    """
    k = sharpness
    # Smooth max(x, low): identity for x >> low, → low for x << low
    x_lo = low + jax.nn.softplus(k * (x - low)) / k
    # Smooth min(x_lo, high): identity for x_lo << high, → high for x_lo >> high
    return high - jax.nn.softplus(k * (high - x_lo)) / k


def trapezoid_cumsum(f: jnp.ndarray, dt: float | jnp.ndarray) -> jnp.ndarray:
    """Trapezoidal cumulative integral with O(dt²) accuracy.

    Computes cumsum[0] = 0, cumsum[k] = Σ_{i=0}^{k-1} (f[i]+f[i+1])/2 × dt.

    This matches homodyne's ``trapezoid_cumsum`` pattern.  The dt factor
    is included in the returned values (unlike homodyne which factors it
    out into the wavevector prefactor).

    Args:
        f: Function values at uniformly spaced time points, shape (N,).
        dt: Time step.

    Returns:
        Cumulative integral, shape (N,).  cumsum[0] = 0 always.
    """
    midpoints = (f[:-1] + f[1:]) / 2.0
    return jnp.concatenate([jnp.zeros(1), jnp.cumsum(midpoints) * dt])


def create_time_integral_matrix(cumsum_values: jnp.ndarray) -> jnp.ndarray:
    """Build N×N integral matrix from cumulative sums (NLSQ meshgrid path).

    M[i,j] = cumsum[j] - cumsum[i]  (signed difference).

    For transport integrals, call ``smooth_abs`` on the result to get
    direction-independent decay.  For velocity integrals, use the signed
    result directly (it feeds into ``cos(q cos(φ) ∫v dt)``).

    Args:
        cumsum_values: Cumulative integral, shape (N,).

    Returns:
        Signed integral matrix, shape (N, N).
    """
    return cumsum_values[None, :] - cumsum_values[:, None]


def compute_transport_rate(
    t: jnp.ndarray,
    D0: float | jnp.ndarray,
    alpha: float | jnp.ndarray,
    offset: float | jnp.ndarray,
) -> jnp.ndarray:
    """Transport rate function J(t) = D0·t^α + offset.

    Shared by both NLSQ and CMC paths — the rate function is the same,
    only the integral evaluation strategy differs.

    Args:
        t: Time array, shape (N,).
        D0: Transport prefactor (Å²/s^α).
        alpha: Transport exponent (dimensionless).
        offset: Constant rate offset (Å²/s).

    Returns:
        Rate values, shape (N,), floored at 0.
    """
    # t_safe: prevent NaN in jnp.power when t=0 with negative alpha.
    # t is a data array (not a parameter), so jnp.where here does not affect
    # the parameter gradient; the outer jnp.where(t > 0) handles t=0 exactly.
    t_safe = jnp.where(t > 1e-10, t, 1e-10)
    t_power = jnp.where(t > 0, jnp.power(t_safe, alpha), 0.0)
    rate = D0 * t_power + offset
    # Physical positivity floor: jnp.maximum is correct here because the
    # subgradient at rate=0 is 1 (gradient of D_offset passes through), while
    # jnp.where(rate > 0.0, rate, 0.0) would block it with strict inequality.
    return jnp.maximum(rate, 0.0)


def compute_velocity_rate(
    t: jnp.ndarray,
    v0: float | jnp.ndarray,
    beta: float | jnp.ndarray,
    v_offset: float | jnp.ndarray,
) -> jnp.ndarray:
    """Velocity rate function v(t) = v0·t^β + v_offset.

    Unlike transport rate, the velocity is NOT floored at 0 because
    the velocity integral enters as cos(q·cos(φ)·∫v dt) which is
    naturally bounded.

    Args:
        t: Time array, shape (N,).
        v0: Velocity prefactor (Å/s^β).
        beta: Velocity exponent (dimensionless).
        v_offset: Constant velocity offset (Å/s).

    Returns:
        Velocity values, shape (N,).
    """
    # Use jnp.where instead of jnp.maximum to preserve gradients below the
    # t=0 floor (jnp.maximum zeros the gradient there).
    t_safe = jnp.where(t > 1e-10, t, 1e-10)
    t_power = jnp.where(t > 0, jnp.power(t_safe, beta), 0.0)
    return v0 * t_power + v_offset


@jax.jit
def safe_sinc(x: jnp.ndarray) -> jnp.ndarray:
    """Unnormalized sinc function sin(x)/x, safe at x=0.

    Returns 1.0 at x=0 (the mathematical limit).

    Args:
        x: Input array (radians, unnormalized).

    Returns:
        sin(x)/x with sinc(0) = 1.
    """
    x = jnp.asarray(x)
    x_safe = jnp.where(jnp.abs(x) > 1e-10, x, 1.0)
    result = jnp.sin(x_safe) / x_safe
    return jnp.where(jnp.abs(x) > 1e-10, result, 1.0)
