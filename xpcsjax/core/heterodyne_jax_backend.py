"""JAX-accelerated computational backend for heterodyne correlation.

This module provides JIT-compiled functions for computing the two-component
heterodyne correlation function using the integral formulation (PNAS Eq. S-95).
All functions are designed to be stateless and compatible with JAX
transformations (jit, vmap, grad).

The correlation is computed as
``c2 = offset + contrast × [ref + sample + cross] / f²``,
where transport terms use the integral of the rate J(t):
``half_tr[i,j] = exp(-½q² × |∫ J_rate(t') dt'|)``.

The 14 model parameters in canonical order are
D0_ref, alpha_ref, D_offset_ref,
D0_sample, alpha_sample, D_offset_sample,
v0, beta, v_offset, f0, f1, f2, f3, phi0.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from xpcsjax.core.heterodyne_physics_utils import (
    compute_transport_rate,
    compute_velocity_rate,
    create_time_integral_matrix,
    safe_exp,
    smooth_abs,
    smooth_clip,
    trapezoid_cumsum,
)

if TYPE_CHECKING:
    pass


@partial(jax.jit, static_argnames=("n_times",))
def compute_transport_jit(
    t: jnp.ndarray,
    D0: float,
    alpha: float,
    offset: float,
    n_times: int,
) -> jnp.ndarray:
    """JIT-compiled pointwise transport coefficient computation.

    J(t) = D0 * t^alpha + offset

    .. deprecated::
        Pointwise approximation — not used in production correlation.
        Production code uses compute_transport_integral_matrix for the
        integral formulation (PNAS Eq. S-95). Retained for test
        compatibility and 1D visualization helpers.

    Args:
        t: Time array
        D0: Transport prefactor
        alpha: Transport exponent
        offset: Constant offset
        n_times: Number of time points (static for JIT)

    Returns:
        Transport coefficient array
    """
    # Use jnp.where instead of jnp.maximum to preserve gradients at the t=0
    # floor (jnp.maximum zeros the gradient when t < 1e-10).
    t_safe = jnp.where(t > 1e-10, t, 1e-10)
    t_power = jnp.power(t_safe, alpha)
    t_power = jnp.where(t > 0, t_power, 0.0)
    return D0 * t_power + offset


@jax.jit
def compute_g1_transport(
    J: jnp.ndarray,
    q: float,
) -> jnp.ndarray:
    """JIT-compiled pointwise g1 correlation from transport coefficient.

    g1(t) = exp(-q² * J(t))

    .. deprecated::
        Pointwise approximation — not used in production correlation.
        Production code uses the integral formulation via
        compute_transport_integral_matrix. Retained for test
        compatibility and 1D visualization helpers.

    Args:
        J: Transport coefficient array
        q: Scattering wavevector

    Returns:
        g1 correlation array
    """
    return jnp.exp(-q * q * J)


@jax.jit
def compute_fraction_jit(
    t: jnp.ndarray,
    f0: float,
    f1: float,
    f2: float,
    f3: float,
) -> jnp.ndarray:
    """JIT-compiled sample fraction computation.

    f_s(t) = f0 * exp(f1 * (t - f2)) + f3, clipped to [0, 1]

    Args:
        t: Time array
        f0: Amplitude
        f1: Exponential rate
        f2: Time shift
        f3: Baseline

    Returns:
        Fraction array in [0, 1]
    """
    # ``safe_exp`` + ``smooth_clip`` preserve gradient at the [0, 1] boundary
    # so NLSQ Jacobian descent does not stall when f(t) saturates (CLAUDE.md
    # rule #7 — gradient-safe floors).
    fraction = f0 * safe_exp(f1 * (t - f2)) + f3
    return smooth_clip(fraction, 0.0, 1.0)


@jax.jit
def compute_velocity_integral_matrix(
    t: jnp.ndarray,
    v0: float,
    beta: float,
    v_offset: float,
    dt: float,
) -> jnp.ndarray:
    """JIT-compiled velocity integral matrix (NLSQ meshgrid path).

    Computes M[i,j] = ∫_{t_i}^{t_j} v(t') dt'
    where v(t) = v0 * t^beta + v_offset

    Uses shared ``trapezoid_cumsum`` → ``create_time_integral_matrix``
    pipeline for O(N) efficiency and O(dt²) accuracy.
    The velocity integral is *signed* (not absolute-valued) because it
    feeds into the phase factor ``cos(q cos(φ) ∫v dt)``.

    Args:
        t: Time array, shape (N,)
        v0: Velocity prefactor
        beta: Velocity exponent
        v_offset: Velocity offset
        dt: Time step

    Returns:
        Signed integral matrix, shape (N, N)
    """
    velocity = compute_velocity_rate(t, v0, beta, v_offset)
    cumsum = trapezoid_cumsum(velocity, dt)
    return create_time_integral_matrix(cumsum)


@jax.jit
def compute_transport_integral_matrix(
    t: jnp.ndarray,
    D0: float,
    alpha: float,
    offset: float,
    dt: float,
) -> jnp.ndarray:
    """JIT-compiled transport integral matrix (NLSQ meshgrid path).

    Computes ``M[i,j] = |∫_{t_i}^{t_j} J_rate(t') dt'|``
    where J_rate(t) = D0 * t^alpha + offset

    Uses shared ``compute_transport_rate`` → ``trapezoid_cumsum`` →
    ``create_time_integral_matrix`` → ``smooth_abs`` pipeline.

    Args:
        t: Time array, shape (N,)
        D0: Transport prefactor
        alpha: Transport exponent
        offset: Transport rate offset
        dt: Time step

    Returns:
        Transport integral matrix, shape (N, N)
    """
    J_rate = compute_transport_rate(t, D0, alpha, offset)
    cumsum = trapezoid_cumsum(J_rate, dt)
    diff = create_time_integral_matrix(cumsum)
    return smooth_abs(diff)


def compute_c2_heterodyne(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    contrast: float = 1.0,
    offset: float = 1.0,
) -> jnp.ndarray:
    """JIT-compiled two-time heterodyne correlation (meshgrid path).

    Thin shim around :func:`xpcsjax.core.heterodyne_physics_kernel.compute_c2_unified`
    with ``eval_strategy="meshgrid"``.  Codex/Gemini G1: the physics math
    lives in the unified kernel; this function preserves the legacy import
    path for all NLSQ call sites (``compute_residuals``, ``compute_jacobian``,
    multi-angle stratified fits, etc.) without behavioural change.

    Args:
        params: Parameter array of shape ``(14,)`` in canonical order:
            ``[D0_ref, alpha_ref, D_offset_ref, D0_sample, alpha_sample,
            D_offset_sample, v0, beta, v_offset, f0, f1, f2, f3, phi0]``.
        t: Time array, shape (N,)
        q: Scattering wavevector magnitude
        dt: Time step
        phi_angle: Detector phi angle (degrees)
        contrast: Speckle contrast (beta), default 1.0
        offset: Baseline offset, default 1.0

    Returns:
        Correlation matrix c2, shape (N, N).
    """
    # Local import to avoid a cycle: physics_kernel.py imports nothing from
    # this module, but importing at module load would chain into JAX init
    # before some downstream test helpers expect it.
    from xpcsjax.core.heterodyne_physics_kernel import compute_c2_unified

    return compute_c2_unified(
        params,
        q,
        dt,
        phi_angle,
        contrast,
        offset,
        eval_strategy="meshgrid",
        t=t,
    )


def compute_residuals(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    contrast: float = 1.0,
    offset: float = 1.0,
) -> jnp.ndarray:
    """Compute weighted residuals between model and data.

    Args:
        params: Parameter array, shape (14,)
        t: Time array
        q: Scattering wavevector
        dt: Time step
        phi_angle: Detector phi angle
        c2_data: Experimental correlation data
        weights: Optional weights (1/uncertainty²)
        contrast: Speckle contrast (beta), default 1.0
        offset: Baseline offset, default 1.0

    Returns:
        Flattened residual array
    """
    if weights is None:
        weights = jnp.ones_like(c2_data)
    return _compute_residuals_jit(  # type: ignore[no-any-return]
        params, t, q, dt, phi_angle, c2_data, weights, contrast, offset
    )


@jax.jit
def _compute_residuals_jit(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    weights: jnp.ndarray,
    contrast: float,
    offset: float,
) -> jnp.ndarray:
    """JIT-compiled residuals computation (always receives weights).

    Diagonal elements (t1==t2) are zeroed per homodyne parity: the diagonal
    is kept in the data for loading/plotting but excluded from fitting because
    corrected diagonal values are interpolated estimates, not real physics.
    """
    c2_model = compute_c2_heterodyne(params, t, q, dt, phi_angle, contrast, offset)
    residuals = (c2_model - c2_data) * jnp.sqrt(weights)
    n_time = c2_data.shape[0]
    non_diagonal = ~jnp.eye(n_time, dtype=bool)
    rows, cols = jnp.nonzero(non_diagonal, size=n_time * (n_time - 1))
    return residuals[rows, cols]  # type: ignore[no-any-return]


# Jacobian of residuals with respect to parameters (for NLSQ).
# jacfwd (forward-mode) does 14 JVP passes for 14 parameters,
# whereas jacobian (reverse-mode) would do ~N² backward passes
# (one per residual element).  For the XPCS use-case with N=200-500
# this is ~8,900x cheaper at N=500 (125K residuals vs 14 params).
_compute_residuals_jacobian_jit = jax.jit(jax.jacfwd(_compute_residuals_jit, argnums=0))


@jax.jit
def compute_chi_squared(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    weights: jnp.ndarray,
    contrast: float,
    offset: float,
) -> jnp.ndarray:
    """JIT-compiled chi-squared computation.

    chi² = sum((c2_model - c2_data)² × weights)

    Args:
        params: Parameter array, shape (14,)
        t: Time array
        q: Scattering wavevector
        dt: Time step
        phi_angle: Detector phi angle
        c2_data: Experimental correlation data
        weights: Weights (1/uncertainty²)
        contrast: Speckle contrast
        offset: Baseline offset

    Returns:
        Chi-squared scalar
    """
    c2_model = compute_c2_heterodyne(params, t, q, dt, phi_angle, contrast, offset)
    return jnp.sum((c2_model - c2_data) ** 2 * weights)


def batch_chi_squared(
    params_batch: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    weights: jnp.ndarray,
    contrast: float = 1.0,
    offset: float = 1.0,
    chunk_size: int | None = None,
) -> jnp.ndarray:
    """Vectorized chi-squared over a batch of parameter sets.

    Uses ``jax.vmap`` for efficient parallel evaluation.  For large batches
    or large time grids, ``chunk_size`` limits simultaneous N×N allocations
    to prevent XLA memory exhaustion (each vmap'd evaluation allocates
    multiple N×N intermediate matrices).

    Args:
        params_batch: Parameter sets, shape ``(n_sets, 14)``.
        t: Time array, shape ``(N,)``.
        q: Scattering wavevector.
        dt: Time step.
        phi_angle: Detector phi angle.
        c2_data: Experimental data.
        weights: Weights.
        contrast: Speckle contrast.
        offset: Baseline offset.
        chunk_size: Max batch elements to vmap simultaneously.  ``None``
            (default) auto-selects based on time-grid size: ``max(1,
            200 // (N // 100))`` to keep peak memory under ~1.6 GB.

    Returns:
        Chi-squared values, shape ``(n_sets,)``.
    """
    n_sets = params_batch.shape[0]
    n_times = t.shape[0]

    def single_chi2(params: jnp.ndarray) -> jnp.ndarray:
        return compute_chi_squared(  # type: ignore[no-any-return]
            params, t, q, dt, phi_angle, c2_data, weights, contrast, offset
        )

    if chunk_size is None:
        # Auto-select: each evaluation creates ~12 N×N float64 matrices
        # (half_tr, cumsum, integral matrix, ref/sample/cross terms, plus
        # XLA intermediates) → ~96 N² bytes per evaluation.
        # Target peak ≈ 1.6 GB → chunk_size ≈ 1.6e9 / (96 * N²).
        matrix_bytes = 96 * n_times * n_times
        chunk_size = max(1, int(1.6e9 / max(matrix_bytes, 1)))

    if n_sets <= chunk_size:
        return jax.vmap(single_chi2)(params_batch)

    # Chunked evaluation to bound peak memory
    chunks = []
    for start in range(0, n_sets, chunk_size):
        chunk = params_batch[start : start + chunk_size]
        chunks.append(jax.vmap(single_chi2)(chunk))
    return jnp.concatenate(chunks)


@jax.jit
def compute_multi_angle_residuals(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angles: jnp.ndarray,
    c2_data_batch: jnp.ndarray,
    weights_batch: jnp.ndarray,
    contrasts: jnp.ndarray,
    offsets: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled residuals for multiple phi angles simultaneously.

    Args:
        params: Parameter array, shape (14,)
        t: Time array, shape (N,)
        q: Scattering wavevector
        dt: Time step
        phi_angles: Phi angles, shape (n_phi,)
        c2_data_batch: Experimental data, shape (n_phi, N, N)
        weights_batch: Weights, shape (n_phi, N, N)
        contrasts: Per-angle contrasts, shape (n_phi,)
        offsets: Per-angle offsets, shape (n_phi,)

    Returns:
        Stacked flattened residuals, shape (n_phi × N × (N-1),)
    """

    def single_angle_residual(
        phi: jnp.ndarray,
        c2_exp: jnp.ndarray,
        w: jnp.ndarray,
        c: jnp.ndarray,
        o: jnp.ndarray,
    ) -> jnp.ndarray:
        c2_model = compute_c2_heterodyne(params, t, q, dt, phi, c, o)
        residuals = (c2_model - c2_exp) * jnp.sqrt(w)
        n_time = c2_exp.shape[0]
        non_diagonal = ~jnp.eye(n_time, dtype=bool)
        rows, cols = jnp.nonzero(non_diagonal, size=n_time * (n_time - 1))
        return residuals[rows, cols]  # type: ignore[no-any-return]

    compute_all = jax.vmap(single_angle_residual, in_axes=(0, 0, 0, 0, 0))
    residuals_batch = compute_all(
        phi_angles, c2_data_batch, weights_batch, contrasts, offsets
    )
    return residuals_batch.ravel()


# Gradient of chi-squared with respect to parameters
compute_chi_squared_grad = jax.jit(jax.grad(compute_chi_squared, argnums=0))

# Hessian of chi-squared (for uncertainty estimation).
# Forward-over-reverse (jacfwd ∘ grad) is preferred over hessian()
# (reverse-over-reverse) for a (14,14) output: it runs 14 JVP passes
# over the gradient graph rather than 14 backward passes over 14
# backward passes, giving a ~14x reduction in graph size on CPU.
compute_chi_squared_hessian = jax.jit(
    jax.jacfwd(jax.grad(compute_chi_squared, argnums=0), argnums=0)
)


def compute_residuals_jacobian(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    contrast: float = 1.0,
    offset: float = 1.0,
) -> jnp.ndarray:
    """Compute Jacobian of residuals with respect to parameters.

    Args:
        params: Parameter array, shape (14,)
        t: Time array
        q: Scattering wavevector
        dt: Time step
        phi_angle: Detector phi angle
        c2_data: Experimental correlation data
        weights: Optional weights (1/uncertainty²)
        contrast: Speckle contrast (beta), default 1.0
        offset: Baseline offset, default 1.0

    Returns:
        Jacobian matrix
    """
    if weights is None:
        weights = jnp.ones_like(c2_data)
    return _compute_residuals_jacobian_jit(  # type: ignore[no-any-return]
        params, t, q, dt, phi_angle, c2_data, weights, contrast, offset
    )
