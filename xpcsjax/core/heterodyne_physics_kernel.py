"""Shared physics kernel for heterodyne two-time correlation.

This module factors the **physics** out of the **evaluation strategy**.
:func:`compute_c2_unified` is one JIT-compiled function whose
``eval_strategy`` static-argument selects:

- ``"meshgrid"``: full N×N two-time matrix (NLSQ path,
  ``compute_residuals``).

The primitives are sourced from
:mod:`xpcsjax.core.heterodyne_physics_utils` (``compute_transport_rate``,
``compute_velocity_rate``, ``trapezoid_cumsum``, ``smooth_abs``,
``smooth_clip``, ``safe_exp``, ``create_signed_integral_matrix``) and apply
gradient-safe clips.

The legacy ``compute_c2_heterodyne`` is retained as a thin shim that
pins ``eval_strategy="meshgrid"`` and forwards.
"""

from __future__ import annotations

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp

from xpcsjax.core.heterodyne_physics_utils import (
    compute_transport_rate,
    compute_velocity_rate,
    create_signed_integral_matrix,
    safe_exp,
    smooth_abs,
    smooth_clip,
    trapezoid_cumsum,
)

EvalStrategy = Literal["meshgrid", "pointwise"]


def _half_transport_meshgrid(
    t: jnp.ndarray,
    D0: jnp.ndarray,
    alpha: jnp.ndarray,
    D_offset: jnp.ndarray,
    q: float,
    dt: float,
) -> jnp.ndarray:
    """Full ``(N, N)`` half-transport matrix: exp(-½ q² |∫J dt|).

    The ``jnp.exp(jnp.clip(...))`` idiom is intentional: clipping the
    exponent at -700 prevents overflow while preserving gradients
    elsewhere.
    """
    rate = compute_transport_rate(t, D0, alpha, D_offset)
    cumsum = trapezoid_cumsum(rate, dt)
    J_integral = smooth_abs(create_signed_integral_matrix(cumsum))
    return jnp.exp(jnp.clip(-0.5 * q * q * J_integral, -700.0, 0.0))


def _velocity_integral_meshgrid(
    t: jnp.ndarray,
    v0: jnp.ndarray,
    beta: jnp.ndarray,
    v_offset: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Full ``(N, N)`` signed velocity integral."""
    velocity = compute_velocity_rate(t, v0, beta, v_offset)
    v_cumsum = trapezoid_cumsum(velocity, dt)
    return create_signed_integral_matrix(v_cumsum)


def _fraction(
    t_vals: jnp.ndarray,
    f0: jnp.ndarray,
    f1: jnp.ndarray,
    f2: jnp.ndarray,
    f3: jnp.ndarray,
) -> jnp.ndarray:
    """Sample-fraction f_s(t), smoothly bounded to [0, 1].

    ``safe_exp`` caps the exponent and ``smooth_clip`` preserves Jacobian
    gradient at the boundary.
    """
    return smooth_clip(f0 * safe_exp(f1 * (t_vals - f2)) + f3, 0.0, 1.0)


@partial(jax.jit, static_argnames=("eval_strategy",))
def compute_c2_unified(
    params: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float = 0.0,
    contrast: float = 1.0,
    offset: float = 1.0,
    *,
    eval_strategy: EvalStrategy = "meshgrid",
    t: jnp.ndarray | None = None,
    # Pointwise-only arguments (all keyword-only)
    phi_unique: jnp.ndarray | None = None,
    phi_idx: jnp.ndarray | None = None,
    t1_idx: jnp.ndarray | None = None,
    t2_idx: jnp.ndarray | None = None,
    contrast_arr: jnp.ndarray | None = None,
    offset_arr: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Two-component heterodyne c2 via the shared kernel.

    Args:
        params: 14-parameter array.
        q: Scattering wavevector magnitude.
        dt: Time step.
        phi_angle: Detector phi angle (degrees) — used by ``"meshgrid"`` only.
        contrast: Speckle contrast (β) — used by ``"meshgrid"`` only.
        offset: Baseline offset — used by ``"meshgrid"`` only.
        eval_strategy: ``"meshgrid"`` for full ``(N, N)`` output;
            ``"pointwise"`` for scattered ``(P,)`` output.
        t: Time array shape ``(N,)`` — required for both strategies.
        phi_unique: Shape ``(n_phi,)`` unique phi angles in degrees —
            required for ``"pointwise"``.
        phi_idx: Shape ``(P,)`` int32 index into ``phi_unique`` per point —
            required for ``"pointwise"``.
        t1_idx: Shape ``(P,)`` int32 time-1 indices — required for
            ``"pointwise"``.
        t2_idx: Shape ``(P,)`` int32 time-2 indices — required for
            ``"pointwise"``.
        contrast_arr: Shape ``(n_phi,)`` per-angle contrast —
            required for ``"pointwise"``.
        offset_arr: Shape ``(n_phi,)`` per-angle offset —
            required for ``"pointwise"``.

    Returns:
        For ``"meshgrid"``: ``(N, N)`` array.
        For ``"pointwise"``: ``(P,)`` array.

    Raises:
        ValueError: Wrong combination of strategy-input arguments.
    """
    if eval_strategy == "meshgrid":
        if t is None:
            raise ValueError(
                "compute_c2_unified(eval_strategy='meshgrid', ...) requires t"
            )
        return _compute_c2_meshgrid(params, t, q, dt, phi_angle, contrast, offset)
    if eval_strategy == "pointwise":
        if t is None:
            raise ValueError(
                "compute_c2_unified(eval_strategy='pointwise', ...) requires t"
            )
        if phi_unique is None or phi_idx is None or t1_idx is None or t2_idx is None:
            raise ValueError(
                "compute_c2_unified(eval_strategy='pointwise', ...) requires "
                "phi_unique, phi_idx, t1_idx, t2_idx"
            )
        if contrast_arr is None or offset_arr is None:
            raise ValueError(
                "compute_c2_unified(eval_strategy='pointwise', ...) requires "
                "contrast_arr and offset_arr"
            )
        return _compute_c2_pointwise(
            params, t, q, dt, phi_unique, phi_idx, t1_idx, t2_idx,
            contrast_arr, offset_arr,
        )
    raise ValueError(
        f"eval_strategy must be 'meshgrid' or 'pointwise', got {eval_strategy!r}"
    )


def _compute_c2_pointwise(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_unique: jnp.ndarray,
    phi_idx: jnp.ndarray,
    t1_idx: jnp.ndarray,
    t2_idx: jnp.ndarray,
    contrast_arr: jnp.ndarray,
    offset_arr: jnp.ndarray,
) -> jnp.ndarray:
    """Pointwise evaluation — produces shape ``(P,)`` from scattered index triples.

    Computes EXACTLY the same physics as ``_compute_c2_meshgrid`` but instead
    of forming full ``(N, N)`` matrices it gathers per-point values using
    ``t1_idx`` / ``t2_idx``.

    The gather identity that guarantees exact parity with the meshgrid branch:
        ``create_signed_integral_matrix(cumsum)[i, j]
          = cumsum[j] - cumsum[i]``
    so pointwise: ``signed_diff = cumsum[t2_idx] - cumsum[t1_idx]``.

    For transport (abs-valued): apply ``smooth_abs`` after gathering.
    For velocity (signed): use the raw difference directly.
    """
    D0_ref, alpha_ref, D_offset_ref = params[0], params[1], params[2]
    D0_sample, alpha_sample, D_offset_sample = params[3], params[4], params[5]
    v0, beta, v_offset = params[6], params[7], params[8]
    f0, f1, f2, f3 = params[9], params[10], params[11], params[12]
    phi0 = params[13]

    # --- Per-t cumulative arrays (computed once over the full time grid) ---
    rate_ref = compute_transport_rate(t, D0_ref, alpha_ref, D_offset_ref)
    cumsum_ref = trapezoid_cumsum(rate_ref, dt)

    rate_sample = compute_transport_rate(t, D0_sample, alpha_sample, D_offset_sample)
    cumsum_sample = trapezoid_cumsum(rate_sample, dt)

    velocity = compute_velocity_rate(t, v0, beta, v_offset)
    v_cumsum = trapezoid_cumsum(velocity, dt)

    f_sample = _fraction(t, f0, f1, f2, f3)  # shape (N,)

    # --- Gather at (t1_idx, t2_idx) pairs ---
    # Transport: signed diff then smooth_abs (mirrors create_signed_integral_matrix + smooth_abs)
    ref_signed = cumsum_ref[t2_idx] - cumsum_ref[t1_idx]
    sample_signed = cumsum_sample[t2_idx] - cumsum_sample[t1_idx]
    J_integral_ref = smooth_abs(ref_signed)
    J_integral_sample = smooth_abs(sample_signed)

    half_tr_ref_t1 = jnp.exp(jnp.clip(-0.5 * q * q * J_integral_ref, -700.0, 0.0))
    half_tr_sample_t1 = jnp.exp(jnp.clip(-0.5 * q * q * J_integral_sample, -700.0, 0.0))

    # Velocity integral: signed (feeds into cos phase)
    v_integral = v_cumsum[t2_idx] - v_cumsum[t1_idx]

    # Fraction at t1 and t2
    fs_t1 = f_sample[t1_idx]
    fs_t2 = f_sample[t2_idx]
    fr_t1 = 1.0 - fs_t1
    fr_t2 = 1.0 - fs_t2

    # --- Per-point phi and cross term ---
    phi_vals = phi_unique[phi_idx]          # shape (P,)
    total_phi = phi_vals + phi0
    phi_rad = jnp.deg2rad(total_phi)
    phase = q * jnp.cos(phi_rad) * v_integral

    # --- Assemble terms (mirrors meshgrid formula exactly) ---
    # ref_term = (fr_t1 * fr_t2)^2 * half_tr_ref^2
    f_ref_prod = fr_t1 * fr_t2
    f_sample_prod = fs_t1 * fs_t2
    f_cross = fr_t1 * fs_t1 * fr_t2 * fs_t2  # (f_ref*f_sample)_t1 * (f_ref*f_sample)_t2

    ref_term = f_ref_prod**2 * half_tr_ref_t1**2
    sample_term = f_sample_prod**2 * half_tr_sample_t1**2
    cross_term = 2.0 * f_cross * half_tr_ref_t1 * half_tr_sample_t1 * jnp.cos(phase)

    norm_t1 = fs_t1**2 + fr_t1**2
    norm_t2 = fs_t2**2 + fr_t2**2
    normalization = norm_t1 * norm_t2

    contrast_p = contrast_arr[phi_idx]
    offset_p = offset_arr[phi_idx]

    return offset_p + contrast_p * (ref_term + sample_term + cross_term) / jnp.where(
        normalization > 1e-10, normalization, 1e-10
    )


def _compute_c2_meshgrid(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    contrast: float,
    offset: float,
) -> jnp.ndarray:
    """Meshgrid evaluation — produces ``(N, N)`` matrix."""
    D0_ref, alpha_ref, D_offset_ref = params[0], params[1], params[2]
    D0_sample, alpha_sample, D_offset_sample = params[3], params[4], params[5]
    v0, beta, v_offset = params[6], params[7], params[8]
    f0, f1, f2, f3 = params[9], params[10], params[11], params[12]
    phi0 = params[13]

    half_tr_ref = _half_transport_meshgrid(t, D0_ref, alpha_ref, D_offset_ref, q, dt)
    half_tr_sample = _half_transport_meshgrid(
        t, D0_sample, alpha_sample, D_offset_sample, q, dt
    )

    f_sample = _fraction(t, f0, f1, f2, f3)
    f_ref = 1.0 - f_sample

    v_integral = _velocity_integral_meshgrid(t, v0, beta, v_offset, dt)

    total_phi = phi_angle + phi0
    phi_rad = jnp.deg2rad(total_phi)
    phase = q * jnp.cos(phi_rad) * v_integral

    f_ref_matrix = f_ref[:, None] * f_ref[None, :]
    f_sample_matrix = f_sample[:, None] * f_sample[None, :]
    f_cross_vec = f_ref * f_sample
    f_cross_matrix = f_cross_vec[:, None] * f_cross_vec[None, :]

    ref_term = f_ref_matrix**2 * half_tr_ref**2
    sample_term = f_sample_matrix**2 * half_tr_sample**2
    cross_term = 2.0 * f_cross_matrix * half_tr_ref * half_tr_sample * jnp.cos(phase)

    norm_1 = f_sample**2 + f_ref**2
    normalization = norm_1[:, None] * norm_1[None, :]

    return offset + contrast * (ref_term + sample_term + cross_term) / jnp.where(
        normalization > 1e-10, normalization, 1e-10
    )
