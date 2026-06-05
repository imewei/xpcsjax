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
    r"""Build the full ``(N, N)`` half-transport matrix.

    Computes ``exp(-0.5 q**2 |\int J dt|)`` over the meshgrid. The
    ``jnp.exp(jnp.clip(...))`` idiom is intentional: clipping the exponent at
    ``-700`` prevents overflow while preserving gradients elsewhere.

    Parameters
    ----------
    t : jnp.ndarray
        Time array, shape ``(N,)``.
    D0 : jnp.ndarray
        Transport prefactor.
    alpha : jnp.ndarray
        Transport exponent.
    D_offset : jnp.ndarray
        Transport rate offset.
    q : float
        Scattering wavevector magnitude.
    dt : float
        Time step.

    Returns
    -------
    jnp.ndarray
        Half-transport matrix, shape ``(N, N)``.
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
    """Build the full ``(N, N)`` signed velocity integral matrix.

    Parameters
    ----------
    t : jnp.ndarray
        Time array, shape ``(N,)``.
    v0 : jnp.ndarray
        Velocity prefactor.
    beta : jnp.ndarray
        Velocity exponent (kernel-internal name for the registry ``v_beta``).
    v_offset : jnp.ndarray
        Velocity offset.
    dt : float
        Time step.

    Returns
    -------
    jnp.ndarray
        Signed velocity-integral matrix, shape ``(N, N)``.
    """
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
    """Compute the sample fraction ``f_s(t)``, smoothly bounded to ``[0, 1]``.

    ``safe_exp`` caps the exponent and ``smooth_clip`` preserves the Jacobian
    gradient at the boundary.

    Parameters
    ----------
    t_vals : jnp.ndarray
        Time array.
    f0 : jnp.ndarray
        Fraction amplitude.
    f1 : jnp.ndarray
        Exponential rate.
    f2 : jnp.ndarray
        Time shift.
    f3 : jnp.ndarray
        Baseline.

    Returns
    -------
    jnp.ndarray
        Sample fraction smoothly clipped to ``[0, 1]``.
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
    """Compute the two-component heterodyne ``c2`` via the shared kernel.

    A single JIT-compiled entry point whose ``eval_strategy`` static argument
    selects between a full meshgrid evaluation and a scattered pointwise
    evaluation; both compute identical physics.

    Parameters
    ----------
    params : jnp.ndarray
        14-parameter array in canonical registry order
        ``[D0_ref, alpha_ref, D_offset_ref, D0_sample, alpha_sample,
        D_offset_sample, v0, v_beta, v_offset, f0, f1, f2, f3, phi0_het]``.
        The kernel uses the internal names ``beta`` / ``phi0`` for the
        registry ``v_beta`` / ``phi0_het``.
    q : float
        Scattering wavevector magnitude.
    dt : float
        Time step.
    phi_angle : float, optional
        Detector phi angle in degrees; used by ``"meshgrid"`` only.
    contrast : float, optional
        Speckle contrast (the prefactor ``beta``); used by ``"meshgrid"``
        only.
    offset : float, optional
        Baseline offset; used by ``"meshgrid"`` only.
    eval_strategy : {"meshgrid", "pointwise"}, optional
        ``"meshgrid"`` for full ``(N, N)`` output; ``"pointwise"`` for
        scattered ``(P,)`` output. Static JIT argument.
    t : jnp.ndarray, optional
        Time array, shape ``(N,)``; required for both strategies.
    phi_unique : jnp.ndarray, optional
        Unique phi angles in degrees, shape ``(n_phi,)``; required for
        ``"pointwise"``.
    phi_idx : jnp.ndarray, optional
        int32 index into ``phi_unique`` per point, shape ``(P,)``; required
        for ``"pointwise"``.
    t1_idx : jnp.ndarray, optional
        int32 time-1 indices, shape ``(P,)``; required for ``"pointwise"``.
    t2_idx : jnp.ndarray, optional
        int32 time-2 indices, shape ``(P,)``; required for ``"pointwise"``.
    contrast_arr : jnp.ndarray, optional
        Per-angle contrast, shape ``(n_phi,)``; required for ``"pointwise"``.
    offset_arr : jnp.ndarray, optional
        Per-angle offset, shape ``(n_phi,)``; required for ``"pointwise"``.

    Returns
    -------
    jnp.ndarray
        For ``"meshgrid"``, an ``(N, N)`` array; for ``"pointwise"``, a
        ``(P,)`` array.

    Raises
    ------
    ValueError
        If the strategy-input arguments are missing or inconsistent with the
        selected ``eval_strategy``.

    Examples
    --------
    >>> c2 = compute_c2_unified(params, q=0.01, dt=0.1, t=t, phi_angle=45.0)
    >>> c2.shape
    (N, N)
    """
    if eval_strategy == "meshgrid":
        if t is None:
            raise ValueError("compute_c2_unified(eval_strategy='meshgrid', ...) requires t")
        return _compute_c2_meshgrid(params, t, q, dt, phi_angle, contrast, offset)
    if eval_strategy == "pointwise":
        if t is None:
            raise ValueError("compute_c2_unified(eval_strategy='pointwise', ...) requires t")
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
            params,
            t,
            q,
            dt,
            phi_unique,
            phi_idx,
            t1_idx,
            t2_idx,
            contrast_arr,
            offset_arr,
        )
    raise ValueError(f"eval_strategy must be 'meshgrid' or 'pointwise', got {eval_strategy!r}")


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
    """Evaluate ``c2`` pointwise, producing ``(P,)`` from scattered index triples.

    Computes exactly the same physics as :func:`_compute_c2_meshgrid` but,
    instead of forming full ``(N, N)`` matrices, gathers per-point values
    using ``t1_idx`` / ``t2_idx``.

    The gather identity that guarantees exact parity with the meshgrid branch
    is ``create_signed_integral_matrix(cumsum)[i, j] = cumsum[j] - cumsum[i]``,
    so pointwise ``signed_diff = cumsum[t2_idx] - cumsum[t1_idx]``. For
    transport (abs-valued) terms ``smooth_abs`` is applied after gathering; for
    velocity (signed) terms the raw difference is used directly.

    Parameters
    ----------
    params : jnp.ndarray
        14-parameter array in canonical registry order.
    t : jnp.ndarray
        Time array, shape ``(N,)``.
    q : float
        Scattering wavevector magnitude.
    dt : float
        Time step.
    phi_unique : jnp.ndarray
        Unique phi angles in degrees, shape ``(n_phi,)``.
    phi_idx : jnp.ndarray
        int32 index into ``phi_unique`` per point, shape ``(P,)``.
    t1_idx : jnp.ndarray
        int32 time-1 indices, shape ``(P,)``.
    t2_idx : jnp.ndarray
        int32 time-2 indices, shape ``(P,)``.
    contrast_arr : jnp.ndarray
        Per-angle contrast, shape ``(n_phi,)``.
    offset_arr : jnp.ndarray
        Per-angle offset, shape ``(n_phi,)``.

    Returns
    -------
    jnp.ndarray
        Flat ``(P,)`` array of ``c2`` values.
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
    phi_vals = phi_unique[phi_idx]  # shape (P,)
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
    """Evaluate ``c2`` over the meshgrid, producing an ``(N, N)`` matrix.

    Parameters
    ----------
    params : jnp.ndarray
        14-parameter array in canonical registry order.
    t : jnp.ndarray
        Time array, shape ``(N,)``.
    q : float
        Scattering wavevector magnitude.
    dt : float
        Time step.
    phi_angle : float
        Detector phi angle in degrees.
    contrast : float
        Speckle contrast (the prefactor ``beta``).
    offset : float
        Baseline offset.

    Returns
    -------
    jnp.ndarray
        Correlation matrix ``c2``, shape ``(N, N)``.
    """
    D0_ref, alpha_ref, D_offset_ref = params[0], params[1], params[2]
    D0_sample, alpha_sample, D_offset_sample = params[3], params[4], params[5]
    v0, beta, v_offset = params[6], params[7], params[8]
    f0, f1, f2, f3 = params[9], params[10], params[11], params[12]
    phi0 = params[13]

    half_tr_ref = _half_transport_meshgrid(t, D0_ref, alpha_ref, D_offset_ref, q, dt)
    half_tr_sample = _half_transport_meshgrid(t, D0_sample, alpha_sample, D_offset_sample, q, dt)

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
