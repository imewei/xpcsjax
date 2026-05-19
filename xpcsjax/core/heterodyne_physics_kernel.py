"""Shared physics kernel for heterodyne two-time correlation.

This module factors the **physics** out of the **evaluation strategy**.
:func:`compute_c2_unified` is one JIT-compiled function whose
``eval_strategy`` static-argument selects:

- ``"meshgrid"``: full N×N two-time matrix (NLSQ path,
  ``compute_residuals``).

The primitives are sourced from
:mod:`xpcsjax.core.heterodyne_physics_utils` (``compute_transport_rate``,
``compute_velocity_rate``, ``trapezoid_cumsum``, ``smooth_abs``,
``smooth_clip``, ``safe_exp``, ``create_time_integral_matrix``) and apply
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
    create_time_integral_matrix,
    safe_exp,
    smooth_abs,
    smooth_clip,
    trapezoid_cumsum,
)

EvalStrategy = Literal["meshgrid"]


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
    J_integral = smooth_abs(create_time_integral_matrix(cumsum))
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
    return create_time_integral_matrix(v_cumsum)


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
    phi_angle: float,
    contrast: float = 1.0,
    offset: float = 1.0,
    *,
    eval_strategy: EvalStrategy = "meshgrid",
    t: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Two-component heterodyne c2 via the shared kernel.

    Args:
        params: 14-parameter array.
        q: Scattering wavevector magnitude.
        dt: Time step.
        phi_angle: Detector phi angle (degrees).
        contrast: Speckle contrast (β).
        offset: Baseline offset.
        eval_strategy: ``"meshgrid"`` for full ``(N, N)`` output (NLSQ).
        t: Time array shape ``(N,)`` — required when
            ``eval_strategy="meshgrid"``.

    Returns:
        For ``eval_strategy="meshgrid"``: ``(N, N)`` array.

    Raises:
        ValueError: Wrong combination of strategy-input arguments.
    """
    if eval_strategy == "meshgrid":
        if t is None:
            raise ValueError(
                "compute_c2_unified(eval_strategy='meshgrid', ...) requires t"
            )
        return _compute_c2_meshgrid(params, t, q, dt, phi_angle, contrast, offset)
    raise ValueError(
        f"eval_strategy must be 'meshgrid', got {eval_strategy!r}"
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
