"""Canonical numerically-safe math primitives shared across physics backends.

This module is the **single source of truth** for math primitives whose
behavior must be identical in the homodyne and heterodyne paths. Historically
``safe_exp`` was defined independently in ``physics_utils.py`` (clip 700) and
``heterodyne_physics_utils.py`` (clip 500); the two silently disagreed for
exponents in (500, 709.78), corrupting heterodyne output at high-q / long-lag
without raising. Consolidating here removes that class of divergence: fix once,
both paths inherit it.

Intentionally **NOT** consolidated here (they differ for sound reasons — do not
naively merge):

- ``safe_sinc``: the homodyne version (``physics_utils.safe_sinc``) uses a
  Taylor expansion ``1 - x²/6 + x⁴/120`` near zero for gradient continuity in
  the Jacobian; the heterodyne version uses a plain guard. Merging would shift
  the homodyne characterization baseline.
- ``trapezoid_cumsum``: homodyne factors ``dt`` out into the wavevector
  prefactor and returns an unscaled cumsum; heterodyne folds ``dt`` into the
  returned values. Different contracts, different signatures.
- the integral matrices: ``physics_utils.create_time_integral_matrix`` takes a
  rate array and returns a smooth-abs'd, zero-diagonal matrix; the heterodyne
  ``create_signed_integral_matrix`` takes a cumsum and returns a raw signed
  difference. Different inputs and different outputs — kept as distinct names.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import jit


@jit
def safe_exp(x: jnp.ndarray | np.ndarray, limit: float = 700.0) -> jnp.ndarray:
    """Overflow-protected exponential, canonical for all physics paths.

    Clips the argument to ``[-limit, limit]`` before ``exp`` to avoid Inf.
    The default ``limit=700`` is the correct float64 bound: ``exp(700) ≈
    1.01e304`` is finite while ``exp(709.79)`` overflows. A smaller clip (the
    old heterodyne value of 500) silently truncates valid exponents in
    ``(500, 709.78)``.

    Parameters
    ----------
    x : jnp.ndarray or np.ndarray
        Input array (NumPy or JAX).
    limit : float, default=700.0
        Symmetric clipping threshold; keep at 700 for float64.

    Returns
    -------
    jnp.ndarray
        ``exp(clip(x, -limit, limit))``, same shape as ``x``.
    """
    x = jnp.asarray(x)
    return jnp.exp(jnp.clip(x, -limit, limit))
