"""Physics-first ⇄ scaling-first optimizer-vector layout conversion.

Phase 2.2. Routing the heterodyne (``two_component``) per-angle fit through the
shared homodyne stratification engine
(:class:`~xpcsjax.optimization.nlsq.strategies.residual_jit.StratifiedResidualFunctionJIT`)
requires reconciling two *different* optimizer parameter-vector layouts. These
helpers are **pure numpy**, run at the optimizer boundary (outside JIT), and do
**no** engine wiring — that is Task 2.3.

Layout A — engine (``StratifiedResidualFunctionJIT``), *scaling-first*
----------------------------------------------------------------------
Verified at ``strategies/residual_jit.py`` ``_compute_single_chunk_residuals``
(lines 304-322). The engine has exactly **two** scaling modes:

* ``use_fixed_scaling=True`` — ``params_all = [physical]`` only. The per-angle
  contrast/offset are frozen on the engine (``fixed_*_per_angle``) and are NOT
  in the optimizer vector.
* ``per_angle_scaling=True`` — ``params_all = [contrast(n_phi) | offset(n_phi) | physical]``.
  Scaling is **EXPANDED**: one contrast and one offset per angle, scaling-first.

The engine has **no compressed averaged mode**. Homodyne's ``auto_averaged``
path broadcasts its 2 averaged scalars to ``n_phi`` *before* the engine via
``data_prep.expand_per_angle_parameters`` (``[contrast, offset, physical]`` →
``[c0..cN-1, o0..oN-1, physical]``). The engine only ever sees the expanded form.

Layout B — heterodyne pointwise (``build_heterodyne_pointwise_model``), *physics-first*
---------------------------------------------------------------------------------------
``strategies/heterodyne_hybrid_streaming.py`` (lines 234-294) builds
``p0 = [physics_varying | scaling_tail]`` where the tail is:

* ``fixed_constant`` — no tail (``n_scaling = 0``). Scaling frozen in the closure.
* ``auto_averaged``  — ``[contrast_scalar, offset_scalar]`` (length 2, COMPRESSED).
* ``individual``     — ``[contrast(n_phi) | offset(n_phi)]`` (length ``2*n_phi``).
* ``fourier``        — Fourier coeffs. **OUT OF SCOPE** here; stays on the
  heterodyne path (a learned reparam, not a permutation).

Reconciliation
--------------
* ``fixed_constant`` — both sides are physics-only. The conversion is the
  IDENTITY on the physics vector.
* ``individual``     — a **pure block permutation**: move the ``2*n_phi`` tail
  from after the physics block to before it. Reversible, covariance-compatible.
* ``auto_averaged``  — **NOT a pure permutation.** The heterodyne side carries 2
  COMPRESSED scalars; the engine side requires ``2*n_phi`` EXPANDED values.
  physics-first → scaling-first must **broadcast** the 2 scalars to ``n_phi``
  each; scaling-first → physics-first must **compress** the (uniform) per-angle
  block back to 2 scalars. A 2-D covariance permutation is therefore undefined
  for ``auto_averaged`` (the dimensionality differs: 2 vs ``2*n_phi``), so
  :func:`scaling_first_permutation` raises for it — covariance must be handled
  on the compressed (physics-first) side for averaged fits.

All vectors are 1-D numpy arrays. ``n_physics`` is the count of *varying*
physics parameters (``len(model.param_manager.varying_names)`` /
``len(get_initial_values())`` — see ``heterodyne_hybrid_streaming.py:234-235``).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "IN_SCOPE_MODES",
    "physics_first_to_scaling_first",
    "scaling_first_to_physics_first",
    "scaling_first_permutation",
    "permute_cov",
]

#: Modes this module converts. ``fourier`` is intentionally excluded — it is a
#: learned reparameterization that stays on the heterodyne path (Task scope).
IN_SCOPE_MODES = ("fixed_constant", "auto_averaged", "individual")


def _validate(mode: str, n_physics: int, n_phi: int) -> None:
    if mode not in IN_SCOPE_MODES:
        raise ValueError(
            f"mode={mode!r} is not an in-scope layout-conversion mode; "
            f"expected one of {IN_SCOPE_MODES}. "
            "('fourier' stays on the heterodyne path and is out of scope.)"
        )
    if n_physics < 0:
        raise ValueError(f"n_physics must be non-negative, got {n_physics}")
    if n_phi < 1:
        raise ValueError(f"n_phi must be >= 1, got {n_phi}")


def _tail_lengths(mode: str, n_phi: int) -> tuple[int, int]:
    """Return ``(physics_first_tail_len, scaling_first_block_len)`` for *mode*.

    ``physics_first_tail_len`` is the heterodyne ``p0`` scaling-tail length;
    ``scaling_first_block_len`` is the engine scaling-block length (always
    expanded ``2*n_phi`` when scaling is in the vector, else 0).
    """
    if mode == "fixed_constant":
        return 0, 0
    if mode == "auto_averaged":
        return 2, 2 * n_phi
    # individual
    return 2 * n_phi, 2 * n_phi


def physics_first_to_scaling_first(
    vec: np.ndarray,
    *,
    n_physics: int,
    mode: str,
    n_phi: int,
) -> np.ndarray:
    """Convert a heterodyne physics-first vector to the engine scaling-first layout.

    Physics-first (input):  ``[physics(n_physics) | scaling_tail]``
    Scaling-first (output): ``[contrast(n_phi) | offset(n_phi) | physics(n_physics)]``
                            (or ``[physics]`` only for ``fixed_constant``).

    For ``auto_averaged`` the 2 compressed scalars are **broadcast** to ``n_phi``
    each (mirrors ``data_prep.expand_per_angle_parameters``). For ``individual``
    this is a pure block permutation. For ``fixed_constant`` it is the identity.
    """
    _validate(mode, n_physics, n_phi)
    vec = np.asarray(vec, dtype=np.float64)
    in_tail, _ = _tail_lengths(mode, n_phi)
    expected_in = n_physics + in_tail
    if vec.shape != (expected_in,):
        raise ValueError(
            f"physics-first vec has shape {vec.shape}; expected ({expected_in},) "
            f"for mode={mode!r}, n_physics={n_physics}, n_phi={n_phi}."
        )

    physics = vec[:n_physics]

    if mode == "fixed_constant":
        return physics.copy()

    if mode == "auto_averaged":
        contrast_scalar = vec[n_physics]
        offset_scalar = vec[n_physics + 1]
        contrast = np.full(n_phi, contrast_scalar, dtype=np.float64)
        offset = np.full(n_phi, offset_scalar, dtype=np.float64)
        return np.concatenate([contrast, offset, physics])

    # individual: tail = [contrast(n_phi) | offset(n_phi)]
    tail = vec[n_physics:]
    contrast = tail[:n_phi]
    offset = tail[n_phi:]
    return np.concatenate([contrast, offset, physics])


def scaling_first_to_physics_first(
    vec: np.ndarray,
    *,
    n_physics: int,
    mode: str,
    n_phi: int,
) -> np.ndarray:
    """Inverse of :func:`physics_first_to_scaling_first`.

    Scaling-first (input):  ``[contrast(n_phi) | offset(n_phi) | physics(n_physics)]``
                            (or ``[physics]`` only for ``fixed_constant``).
    Physics-first (output): ``[physics(n_physics) | scaling_tail]``.

    For ``auto_averaged`` the (uniform) per-angle blocks are **compressed** back
    to their 2 scalars by taking the first element of each block — exact round
    trip because the forward map broadcasts a single value. (No mean is taken:
    the forward map guarantees uniformity, and element-0 is the exact inverse.)
    """
    _validate(mode, n_physics, n_phi)
    vec = np.asarray(vec, dtype=np.float64)
    _, block = _tail_lengths(mode, n_phi)
    expected_in = block + n_physics
    if vec.shape != (expected_in,):
        raise ValueError(
            f"scaling-first vec has shape {vec.shape}; expected ({expected_in},) "
            f"for mode={mode!r}, n_physics={n_physics}, n_phi={n_phi}."
        )

    if mode == "fixed_constant":
        return vec.copy()

    contrast = vec[:n_phi]
    offset = vec[n_phi : 2 * n_phi]
    physics = vec[2 * n_phi :]

    if mode == "auto_averaged":
        # Forward broadcast guarantees uniformity; element-0 is the exact inverse.
        return np.concatenate([physics, contrast[:1], offset[:1]])

    # individual: physics-first tail = [contrast(n_phi) | offset(n_phi)]
    return np.concatenate([physics, contrast, offset])


def scaling_first_permutation(
    *,
    n_physics: int,
    mode: str,
    n_phi: int,
) -> np.ndarray:
    """Index permutation taking a physics-first vector to the scaling-first layout.

    Returns ``perm`` such that ``scaling_first_vec == physics_first_vec[perm]``
    for the **pure-permutation** modes (``fixed_constant``, ``individual``).
    This same ``perm`` drives the 2-D covariance permutation via
    :func:`permute_cov`.

    Raises ``ValueError`` for ``auto_averaged`` — that mode is a
    broadcast/compress, not a permutation, so there is no single index map and a
    2-D covariance permutation is undefined (dimensions 2 vs ``2*n_phi`` differ).
    Handle averaged-mode covariance on the compressed physics-first side.
    """
    _validate(mode, n_physics, n_phi)
    if mode == "auto_averaged":
        raise ValueError(
            "auto_averaged is a broadcast/compress, not a permutation; no index "
            "map exists (physics-first tail length 2 != scaling-first block "
            f"length {2 * n_phi}). Permute covariance on the physics-first side."
        )

    if mode == "fixed_constant":
        # Physics-only on both sides: identity.
        return np.arange(n_physics, dtype=np.int64)

    # individual:
    #   physics-first indices: [0..n_physics)          physics
    #                          [n_physics..+n_phi)     contrast
    #                          [+n_phi..+2*n_phi)      offset
    #   scaling-first target:  [contrast | offset | physics]
    # perm[j] = physics-first index that lands at scaling-first position j.
    contrast_src = np.arange(n_physics, n_physics + n_phi, dtype=np.int64)
    offset_src = np.arange(n_physics + n_phi, n_physics + 2 * n_phi, dtype=np.int64)
    physics_src = np.arange(0, n_physics, dtype=np.int64)
    return np.concatenate([contrast_src, offset_src, physics_src])


def permute_cov(P: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Apply a 2-D block permutation to a covariance matrix.

    Permutes BOTH rows and columns: ``P[np.ix_(perm, perm)]``. This is the
    correct transform for a covariance/Hessian under a coordinate permutation
    (a 1-D reorder of rows alone would not preserve the quadratic form).

    Symmetry is preserved, and the operation is its own inverse under the
    inverse permutation: ``permute_cov(permute_cov(P, perm), argsort(perm)) == P``.
    """
    P = np.asarray(P, dtype=np.float64)
    perm = np.asarray(perm, dtype=np.int64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"P must be a square 2-D matrix, got shape {P.shape}")
    if perm.shape != (P.shape[0],):
        raise ValueError(
            f"perm length {perm.shape} does not match covariance dimension {P.shape[0]}"
        )
    return P[np.ix_(perm, perm)]
