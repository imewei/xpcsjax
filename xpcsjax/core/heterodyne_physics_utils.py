"""Numerically safe mathematical primitives for heterodyne physics.

All functions are designed to be compatible with both NumPy and JAX
arrays, avoiding NaN/Inf from edge cases (division by zero,
overflow in exp, negative bases in power).

Shared utilities used by the NLSQ meshgrid path:
- ``trapezoid_cumsum``: O(dt²) cumulative integral (dt folded IN — differs from
  the homodyne ``physics_utils.trapezoid_cumsum`` which factors dt out; do not
  merge the two)
- ``create_signed_integral_matrix``: N×N signed difference from cumsum (NLSQ
  only). Distinct from ``physics_utils.create_time_integral_matrix`` (which
  takes a rate, integrates internally, and smooth-abs'es the result).
- ``smooth_abs``: gradient-safe ``|x|``
- ``compute_transport_rate``: J(t) = D0·t^α + offset
- ``compute_velocity_rate``: v(t) = v0·t^β + v_offset

``safe_exp`` is re-exported from ``math_primitives`` (canonical, clip 700).
``safe_sinc`` here intentionally differs from the homodyne Taylor-expansion
version; see ``math_primitives`` module docstring.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# safe_exp is canonical in math_primitives (clip 700, the correct float64 bound).
# It previously lived here with a clip of 500, which silently truncated valid
# exponents in (500, 709.78). Re-exported for backward compatibility.
from xpcsjax.core.math_primitives import safe_exp  # noqa: F401

# ---------------------------------------------------------------------------
# Numerically safe math primitives
# ---------------------------------------------------------------------------


def safe_power(base: jnp.ndarray | np.ndarray, exponent: float) -> jnp.ndarray:
    """Compute a power function that is safe for non-positive bases.

    For ``base <= 0`` this returns ``0.0`` (the physical limit for the
    ``t**alpha`` transport term); for ``base > 0`` it returns
    ``base ** exponent`` normally.

    Parameters
    ----------
    base : jnp.ndarray or np.ndarray
        Base array, typically time values.
    exponent : float
        Power exponent.

    Returns
    -------
    jnp.ndarray
        Safe power result, same shape as ``base``.
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
    """Divide with protection against zero or near-zero denominators.

    Parameters
    ----------
    numerator : jnp.ndarray or np.ndarray
        Dividend array.
    denominator : jnp.ndarray or np.ndarray
        Divisor array.
    fill : float, optional
        Value to return where the denominator is too small.
    min_denom : float, optional
        Minimum absolute denominator magnitude treated as non-zero.

    Returns
    -------
    jnp.ndarray
        Safe quotient, broadcast to the shape of the inputs.
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
    """Compute a logarithm with protection against non-positive arguments.

    Parameters
    ----------
    x : jnp.ndarray or np.ndarray
        Input array.
    floor : float, optional
        Minimum value substituted before taking the logarithm.

    Returns
    -------
    jnp.ndarray
        ``log(max(x, floor))``, same shape as ``x``.
    """
    x = jnp.asarray(x)
    # Use jnp.where to preserve gradients: jnp.maximum zeros the gradient
    # when x < floor, stalling log-space parameter updates.
    return jnp.log(jnp.where(x > floor, x, floor))


def safe_sqrt(x: jnp.ndarray | np.ndarray) -> jnp.ndarray:
    """Compute a square root with protection against negative arguments.

    Parameters
    ----------
    x : jnp.ndarray or np.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        ``sqrt(max(x, 0))``, same shape as ``x``.
    """
    x = jnp.asarray(x)
    # Use jnp.where to preserve gradients: jnp.maximum zeros the gradient
    # when x < 0, which would stall the Jacobian at the sqrt floor.
    return jnp.sqrt(jnp.where(x > 0.0, x, 0.0))


def compute_relative_difference(
    a: jnp.ndarray | np.ndarray,
    b: jnp.ndarray | np.ndarray,
) -> jnp.ndarray:
    """Compute the element-wise relative difference between two arrays.

    Defined as ``|a - b| / max(|a|, |b|, 1e-10)``. Useful for comparing
    correlation matrices or parameter arrays where absolute differences may
    mislead across different scales.

    Parameters
    ----------
    a : jnp.ndarray or np.ndarray
        First array.
    b : jnp.ndarray or np.ndarray
        Second array.

    Returns
    -------
    jnp.ndarray
        Relative-difference array with values in ``[0, 2]``.
    """
    a, b = jnp.asarray(a), jnp.asarray(b)
    max_abs = jnp.maximum(jnp.maximum(jnp.abs(a), jnp.abs(b)), 1e-10)
    return jnp.abs(a - b) / max_abs


def symmetrize(matrix: jnp.ndarray | np.ndarray) -> jnp.ndarray:
    """Force a matrix to be exactly symmetric via ``(M + M.T) / 2``.

    Parameters
    ----------
    matrix : jnp.ndarray or np.ndarray
        Square matrix.

    Returns
    -------
    jnp.ndarray
        Symmetrized matrix.
    """
    m = jnp.asarray(matrix)
    return 0.5 * (m + m.T)


# ---------------------------------------------------------------------------
# Shared integral and rate primitives (used by the NLSQ path)
# ---------------------------------------------------------------------------


def smooth_abs(x: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Compute a gradient-safe absolute value ``sqrt(x**2 + eps)``.

    ``jnp.abs(x)`` has an undefined gradient at ``x = 0``, which produces
    NaN gradients on matrix diagonals where integrals vanish. This smooth
    approximation matches ``|x|`` to ``O(sqrt(eps))`` and has well-defined
    gradients everywhere.

    Parameters
    ----------
    x : jnp.ndarray
        Input array.
    eps : float, optional
        Smoothing parameter; ``1e-12`` gives a ~``1e-6`` bias on the diagonal.

    Returns
    -------
    jnp.ndarray
        Smooth ``|x|``, same shape as ``x``.

    Notes
    -----
    Although named ``safe_exp`` is canonical elsewhere, this is a heterodyne-
    specific gradient-safe primitive. The bias is the price paid for a finite
    gradient at the origin.
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

    The boundary smear scales as ``1 / sharpness`` — at the default value
    ``sharpness=50`` the boundary lands within ~``(high - low) / 50`` of the
    target (about 2% of the range), with monotonic identity in the interior.
    Raise ``sharpness`` for a tighter approximation at the cost of gradient
    magnitude near the boundary; lower it for stronger regularisation.

    Parameters
    ----------
    x : jnp.ndarray
        Input array of any shape.
    low : float
        Lower physical bound (inclusive in the limit).
    high : float
        Upper physical bound (inclusive in the limit).
    sharpness : float, optional
        Softplus sharpness; default ``50`` gives a ~2% boundary smear.

    Returns
    -------
    jnp.ndarray
        Smoothly bounded array, asymptotically within ``(low, high)``, with
        well-defined gradients everywhere.
    """
    k = sharpness
    # Smooth max(x, low): identity for x >> low, → low for x << low
    x_lo = low + jax.nn.softplus(k * (x - low)) / k
    # Smooth min(x_lo, high): identity for x_lo << high, → high for x_lo >> high
    return high - jax.nn.softplus(k * (high - x_lo)) / k


def trapezoid_cumsum(f: jnp.ndarray, dt: float | jnp.ndarray) -> jnp.ndarray:
    """Compute a trapezoidal cumulative integral with ``O(dt**2)`` accuracy.

    Yields ``cumsum[0] = 0`` and
    ``cumsum[k] = sum_{i=0}^{k-1} (f[i] + f[i+1]) / 2 * dt``.

    This matches the homodyne ``trapezoid_cumsum`` pattern, except the ``dt``
    factor is folded into the returned values here (homodyne factors it out
    into the wavevector prefactor); do not merge the two implementations.

    Parameters
    ----------
    f : jnp.ndarray
        Function values at uniformly spaced time points, shape ``(N,)``.
    dt : float or jnp.ndarray
        Time step.

    Returns
    -------
    jnp.ndarray
        Cumulative integral, shape ``(N,)``; ``cumsum[0]`` is always ``0``.
    """
    midpoints = (f[:-1] + f[1:]) / 2.0
    return jnp.concatenate([jnp.zeros(1), jnp.cumsum(midpoints) * dt])


def create_signed_integral_matrix(cumsum_values: jnp.ndarray) -> jnp.ndarray:
    """Build N×N **signed** integral matrix from cumulative sums (NLSQ path).

    M[i,j] = cumsum[j] - cumsum[i]  (signed difference).

    NOTE: This is deliberately NOT the same function as
    ``physics_utils.create_time_integral_matrix``. The homodyne version takes a
    *rate* array, integrates it internally, and returns a smooth-abs'd matrix
    with a zeroed diagonal. This one takes an already-integrated *cumsum* and
    returns a raw signed difference. The two shared a name historically, which
    let a homodyne-contract caller silently receive signed (wrong-sign) decay
    values — hence the rename.

    For transport integrals, call ``smooth_abs`` on the result to get
    direction-independent decay. For velocity integrals, use the signed
    result directly (it feeds into ``cos(q cos(phi) integral(v) dt)``).

    Parameters
    ----------
    cumsum_values : jnp.ndarray
        Cumulative integral, shape ``(N,)``.

    Returns
    -------
    jnp.ndarray
        Signed integral matrix ``M[i, j] = cumsum[j] - cumsum[i]``,
        shape ``(N, N)``.

    See Also
    --------
    smooth_abs : Gradient-safe absolute value applied to transport integrals.
    """
    return cumsum_values[None, :] - cumsum_values[:, None]


def compute_transport_rate(
    t: jnp.ndarray,
    D0: float | jnp.ndarray,
    alpha: float | jnp.ndarray,
    offset: float | jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the transport rate ``J(t) = D0 * t**alpha + offset``.

    Used by the NLSQ path for integral evaluation.

    Parameters
    ----------
    t : jnp.ndarray
        Time array, shape ``(N,)``.
    D0 : float or jnp.ndarray
        Transport prefactor in ``Angstrom**2 / s**alpha``.
    alpha : float or jnp.ndarray
        Transport exponent (dimensionless).
    offset : float or jnp.ndarray
        Constant rate offset in ``Angstrom**2 / s``.

    Returns
    -------
    jnp.ndarray
        Rate values, shape ``(N,)``, floored at ``0``.
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
    """Evaluate the velocity rate ``v(t) = v0 * t**beta + v_offset``.

    Unlike the transport rate, the velocity is not floored at ``0`` because
    the velocity integral enters as ``cos(q cos(phi) integral(v) dt)``, which
    is naturally bounded.

    Parameters
    ----------
    t : jnp.ndarray
        Time array, shape ``(N,)``.
    v0 : float or jnp.ndarray
        Velocity prefactor in ``Angstrom / s**beta``.
    beta : float or jnp.ndarray
        Velocity exponent (dimensionless). This is the kernel-internal name
        for the registry parameter ``v_beta``.
    v_offset : float or jnp.ndarray
        Constant velocity offset in ``Angstrom / s``.

    Returns
    -------
    jnp.ndarray
        Velocity values, shape ``(N,)``.
    """
    # Use jnp.where instead of jnp.maximum to preserve gradients below the
    # t=0 floor (jnp.maximum zeros the gradient there).
    t_safe = jnp.where(t > 1e-10, t, 1e-10)
    t_power = jnp.where(t > 0, jnp.power(t_safe, beta), 0.0)
    return v0 * t_power + v_offset


@jax.jit
def safe_sinc(x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the unnormalized sinc ``sin(x) / x``, safe at ``x = 0``.

    Returns ``1.0`` at ``x = 0`` (the mathematical limit). This intentionally
    differs from the homodyne Taylor-expansion variant; see the
    ``math_primitives`` module docstring.

    Parameters
    ----------
    x : jnp.ndarray
        Input array in radians (unnormalized).

    Returns
    -------
    jnp.ndarray
        ``sin(x) / x`` with ``sinc(0) = 1``.
    """
    x = jnp.asarray(x)
    x_safe = jnp.where(jnp.abs(x) > 1e-10, x, 1.0)
    result = jnp.sin(x_safe) / x_safe
    return jnp.where(jnp.abs(x) > 1e-10, result, 1.0)
