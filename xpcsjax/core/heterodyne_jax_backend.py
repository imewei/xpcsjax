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

from functools import lru_cache
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

# Physics primitives are sourced from ``heterodyne_physics_utils`` (NOT the
# homodyne ``physics_utils``): the two define same-named helpers with different
# contracts (e.g. ``create_signed_integral_matrix`` here returns a *signed*
# difference, while ``physics_utils.create_time_integral_matrix`` returns a
# smooth-abs'd matrix). ``safe_exp`` is the shared canonical one re-exported
# from ``math_primitives``.
from xpcsjax.core.heterodyne_physics_utils import (
    compute_transport_rate,
    compute_velocity_rate,
    create_signed_integral_matrix,
    safe_exp,
    smooth_abs,
    smooth_clip,
    trapezoid_cumsum,
)

if TYPE_CHECKING:
    pass



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

    Uses shared ``trapezoid_cumsum`` → ``create_signed_integral_matrix``
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
    return create_signed_integral_matrix(cumsum)


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
    ``create_signed_integral_matrix`` → ``smooth_abs`` pipeline.

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
    diff = create_signed_integral_matrix(cumsum)
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


def compute_c2_heterodyne_pointwise(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    *,
    phi_unique: jnp.ndarray,
    phi_idx: jnp.ndarray,
    t1_idx: jnp.ndarray,
    t2_idx: jnp.ndarray,
    contrast: jnp.ndarray,
    offset: jnp.ndarray,
) -> jnp.ndarray:
    """Pointwise heterodyne correlation at scattered ``(phi, t1, t2)`` triples.

    Thin shim around :func:`xpcsjax.core.heterodyne_physics_kernel.compute_c2_unified`
    with ``eval_strategy="pointwise"``.  Returns a flat ``(P,)`` array —
    one c2 value per point — suitable for feeding NLSQ's
    ``AdaptiveHybridStreamingOptimizer``.

    Physics is IDENTICAL to the meshgrid path: per-t cumulative arrays are
    computed once and then gathered at ``(t1_idx[p], t2_idx[p])`` per point,
    giving exact float-level parity with
    ``compute_c2_heterodyne(params, t, q, dt, phi_unique[phi_idx[p]],
      contrast[phi_idx[p]], offset[phi_idx[p]])[t1_idx[p], t2_idx[p]]``.

    Args:
        params: Parameter array of shape ``(14,)`` in canonical order.
        t: Time array, shape ``(N,)``.
        q: Scattering wavevector magnitude.
        dt: Time step.
        phi_unique: Shape ``(n_phi,)`` unique phi angles in degrees.
        phi_idx: Shape ``(P,)`` int32 index into ``phi_unique`` per point.
        t1_idx: Shape ``(P,)`` int32 time-1 indices into ``t``.
        t2_idx: Shape ``(P,)`` int32 time-2 indices into ``t``.
        contrast: Shape ``(n_phi,)`` per-angle speckle contrast.
        offset: Shape ``(n_phi,)`` per-angle baseline offset.

    Returns:
        Flat ``(P,)`` array of c2 values.
    """
    from xpcsjax.core.heterodyne_physics_kernel import compute_c2_unified

    return compute_c2_unified(
        params,
        q,
        dt,
        eval_strategy="pointwise",
        t=t,
        phi_unique=phi_unique,
        phi_idx=phi_idx,
        t1_idx=t1_idx,
        t2_idx=t2_idx,
        contrast_arr=contrast,
        offset_arr=offset,
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


@lru_cache(maxsize=64)
def _offdiag_indices(n_time: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the static off-diagonal gather indices for an ``n_time × n_time`` grid.

    The chi-square support excludes the t=0 row/column and the ``t1==t2``
    diagonal, leaving ``(n_time-1) * (n_time-2)`` pairs where both row > 0 and
    col > 0. This set depends only on ``n_time`` (a concrete int at trace time),
    so it is computed once host-side with NumPy and memoized — avoiding the
    per-evaluation ``jnp.eye`` + ``jnp.nonzero`` (~30x slower) that XLA does not
    constant-fold (constant folding is disabled package-wide).

    The arrays are returned as plain NumPy (not ``jnp.asarray``): JAX accepts
    NumPy integer index arrays in ``residuals[rows, cols]`` and bakes them in as
    trace constants. Materializing ``jnp`` arrays here would, on a first call
    made *inside* a trace, capture that trace into the memoized value and leak it
    across subsequent traces (UnexpectedTracerError under vmap).
    """
    indices = np.arange(n_time)
    boundary_mask = (indices[:, None] > 0) & (indices[None, :] > 0)
    valid_mask = boundary_mask & ~np.eye(n_time, dtype=bool)
    rows, cols = np.nonzero(valid_mask)
    return rows, cols


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

    Two boundary exclusions are applied at residual construction (upstream
    ``heterodyne._compute_residuals_jit`` parity):

    1. The t=0 row ``(0, j)`` and t=0 column ``(i, 0)`` are excluded. The
       first frame holds the correlator's raw output; the experimental
       boundary is not used in chi-square fitting.
    2. The diagonal ``t1==t2`` is excluded (homodyne parity): corrected
       diagonal values are interpolated estimates, not real physics.

    The returned vector has shape ``(n_time-1) * (n_time-2)`` — only
    off-diagonal pairs where both row > 0 and col > 0.
    """
    c2_model = compute_c2_heterodyne(params, t, q, dt, phi_angle, contrast, offset)
    residuals = (c2_model - c2_data) * jnp.sqrt(weights)
    rows, cols = _offdiag_indices(c2_data.shape[0])
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
        Stacked flattened residuals, shape (n_phi × (N-1) × (N-2),)
    """

    def single_angle_residual(
        phi: jnp.ndarray,
        c2_exp: jnp.ndarray,
        w: jnp.ndarray,
        c: jnp.ndarray,
        o: jnp.ndarray,
    ) -> jnp.ndarray:
        # Match _compute_residuals_jit / upstream: exclude BOTH the t=0
        # boundary row/col and the diagonal before flattening, so joint
        # multi-phi fits use the same chi-square support as single-phi fits.
        c2_model = compute_c2_heterodyne(params, t, q, dt, phi, c, o)
        residuals = (c2_model - c2_exp) * jnp.sqrt(w)
        rows, cols = _offdiag_indices(c2_exp.shape[0])
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
