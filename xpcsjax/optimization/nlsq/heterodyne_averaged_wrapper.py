"""Compressed-averaged residual wrapper for engine routing (Task #14).

Routing the heterodyne (``two_component``) ``auto_averaged`` per-angle fit through
the shared homodyne stratification engine
(:class:`~xpcsjax.optimization.nlsq.strategies.residual_jit.StratifiedResidualFunctionJIT`)
is *not* apples-to-apples with production unless the optimizer varies the same
number of scaling DOF.

* **Production** (``_fit_joint_averaged_multi_phi``) optimizes a **compressed**
  ``[physics | avg_contrast, avg_offset]`` vector — exactly **2 scaling DOF**,
  broadcast uniformly across every angle.
* **The engine** with ``per_angle_scaling=True`` exposes ``2*n_phi`` *independent*
  scaling params (``[contrast(n_phi) | offset(n_phi) | physics]``, scaling-first).
  Driving the engine directly therefore over-parameterizes the averaged fit by
  ``2*(n_phi-1)`` scaling DOF — not the production averaged problem.

This module closes that gap with a thin **JAX-native** residual wrapper. The
optimizer varies only the 2 averaged scalars (physics-first compressed,
``[physics(n_physics) | c_avg | o_avg]``); *inside* the JIT-traced residual the
wrapper broadcasts those 2 scalars to the engine's ``2*n_phi`` scaling-first
layout and calls the engine residual. The optimizer therefore sees an
``(n_physics + 2)``-param problem — exactly the production averaged DOF count —
while the engine still evaluates its per-angle scaling-first surface.

This is a **heterodyne-side wrapper, NOT an engine change.** The engine's
``__call__`` is untouched; the homodyne preservation suite stays bit-identical.

Layouts (mirror :mod:`xpcsjax.optimization.nlsq.heterodyne_layout`)
-------------------------------------------------------------------
* compressed physics-first (optimizer-facing): ``[physics(n_physics) | c_avg | o_avg]``
  — length ``n_physics + 2``.
* engine scaling-first (per-angle):
  ``[contrast(n_phi) | offset(n_phi) | physics(n_physics)]`` — length
  ``2*n_phi + n_physics``.

The broadcast is a ``jnp.full((n_phi,), c_avg)`` / ``jnp.full((n_phi,), o_avg)``
performed on the live tracer — JIT- and autodiff-safe. The boundary conversions
(x0/bounds in, popt out) are pure numpy and run *outside* the traced residual.
Covariance for the averaged fit is dimensionally on the **compressed** side (2
scaling DOF), so it is passed through unchanged — consistent with the Task-2.2
finding that an averaged 2-D covariance permutation onto ``2*n_phi`` is undefined
(``heterodyne_layout.scaling_first_permutation`` raises for ``auto_averaged``).
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

__all__ = [
    "wrap_engine_averaged_residual",
    "compressed_averaged_to_engine_scaling_first",
    "engine_popt_to_compressed_averaged",
]


def _validate(n_physics: int, n_phi: int) -> None:
    if n_physics < 0:
        raise ValueError(f"n_physics must be non-negative, got {n_physics}")
    if n_phi < 1:
        raise ValueError(f"n_phi must be >= 1, got {n_phi}")


def compressed_averaged_to_engine_scaling_first(
    x: jnp.ndarray | np.ndarray,
    *,
    n_physics: int,
    n_phi: int,
) -> jnp.ndarray:
    """Broadcast a compressed averaged vector to the engine scaling-first layout.

    Input (compressed physics-first): ``[physics(n_physics) | c_avg | o_avg]``.
    Output (engine scaling-first):
    ``[contrast(n_phi) | offset(n_phi) | physics(n_physics)]`` with the 2 averaged
    scalars broadcast uniformly across all ``n_phi`` angles.

    JAX-native: the broadcast uses ``jnp.full`` on the (possibly traced) input, so
    this is safe to call *inside* a JIT-traced residual. (The numpy boundary
    helpers below produce the same mapping for x0/bounds outside the trace.)
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    physics = x[:n_physics]
    c_avg = x[n_physics]
    o_avg = x[n_physics + 1]
    contrast = jnp.full((n_phi,), c_avg, dtype=jnp.float64)
    offset = jnp.full((n_phi,), o_avg, dtype=jnp.float64)
    return jnp.concatenate([contrast, offset, physics])


def wrap_engine_averaged_residual(
    engine: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    n_physics: int,
    n_phi: int,
) -> Callable[[np.ndarray | jnp.ndarray], jnp.ndarray]:
    """Wrap a ``per_angle_scaling=True`` engine residual for compressed averaged mode.

    The returned callable takes a compressed physics-first vector
    ``x = [physics(n_physics) | c_avg | o_avg]`` (length ``n_physics + 2``) and
    returns the engine residuals. Inside (JAX, trace-safe) it broadcasts the 2
    averaged scalars to the engine's ``2*n_phi`` scaling-first layout via
    :func:`compressed_averaged_to_engine_scaling_first` and calls ``engine``.

    The optimizer (``least_squares``) therefore solves an ``(n_physics + 2)``-param
    problem — exactly the production averaged scaling DOF — even though the engine
    still evaluates its per-angle scaling surface internally.

    Args:
        engine: A residual callable built with ``per_angle_scaling=True`` (e.g.
            :class:`StratifiedResidualFunctionJIT`). It must accept the engine
            scaling-first vector ``[contrast(n_phi) | offset(n_phi) | physics]``.
        n_physics: Number of varying physics parameters.
        n_phi: Number of angles (engine per-angle scaling width).
    """
    _validate(n_physics, n_phi)

    def residual(x: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        engine_vec = compressed_averaged_to_engine_scaling_first(
            x, n_physics=n_physics, n_phi=n_phi
        )
        return engine(engine_vec)

    return residual


def engine_popt_to_compressed_averaged(
    x: np.ndarray | jnp.ndarray,
    *,
    n_physics: int,
) -> np.ndarray:
    """Identity passthrough for the optimized compressed-averaged popt (boundary).

    Because the wrapper makes the optimizer vary the compressed
    ``[physics | c_avg | o_avg]`` vector directly, the optimized ``popt`` is
    *already* physics-first compressed — no un-permutation is needed. This helper
    exists for symmetry with the ``heterodyne_layout`` boundary API and validates
    the length. Covariance stays on this (compressed) side per the Task-2.2
    finding (an averaged 2->2*n_phi covariance permutation is undefined).
    """
    x = np.asarray(x, dtype=np.float64)
    expected = n_physics + 2
    if x.shape != (expected,):
        raise ValueError(
            f"compressed-averaged popt has shape {x.shape}; expected ({expected},) "
            f"for n_physics={n_physics} (physics-first [physics | c_avg | o_avg])."
        )
    return x.copy()
