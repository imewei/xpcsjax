"""Test-local oracle for the heterodyne physics-first <-> scaling-first layout.

The production ``heterodyne_layout`` module was retired (Task 11) once the engine
route began building its scaling-first x0 directly. The engine-parity tests still
need the physics-first <-> scaling-first conversion as an INDEPENDENT reference
(an oracle the engine route's inline build is checked against), so the pure-math
conversion is preserved here, in the test tree only — it is not production code.

The functions are byte-for-byte equivalent to the former
``heterodyne_layout.{physics_first_to_scaling_first,scaling_first_to_physics_first}``
and ``IN_SCOPE_MODES``.
"""

from __future__ import annotations

import numpy as np

#: The three in-scope per-angle scaling layout-conversion modes (OLD tokens).
IN_SCOPE_MODES = ("fixed_constant", "auto_averaged", "individual")


def _validate(mode: str, n_physics: int, n_phi: int) -> None:
    if mode not in IN_SCOPE_MODES:
        raise ValueError(
            f"mode={mode!r} is not an in-scope layout-conversion mode; "
            f"expected one of {IN_SCOPE_MODES}."
        )
    if n_physics < 0:
        raise ValueError(f"n_physics must be non-negative, got {n_physics}")
    if n_phi < 1:
        raise ValueError(f"n_phi must be >= 1, got {n_phi}")


def _tail_lengths(mode: str, n_phi: int) -> tuple[int, int]:
    """Return ``(physics_first_tail_len, scaling_first_block_len)`` for *mode*."""
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
