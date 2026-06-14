"""Test-local oracle for the canonical heterodyne scaling-first layout.

The production layout module was retired (Task 11) once the engine
route began building its scaling-first x0 directly, AND Phase 4 made the pointwise
builder emit the canonical **scaling-first** ``p0`` natively
(``[scaling_head | physics_tail]``). The former physics-first<->scaling-first
conversion is therefore the *identity* for ``constant`` / ``individual`` (their
builder head already equals the engine's scaling-first layout) and a pure
*broadcast-expand* for ``averaged`` (the builder emits a compressed 2-scalar head
``[c_avg, o_avg]`` that the engine's ``per_angle_scaling=True`` path needs widened
to ``2 * n_phi``).

The engine-parity tests still need this expansion as an INDEPENDENT reference (an
oracle the engine route's inline build is checked against), so the pure-math
``expand_to_engine_scaling_first`` is preserved here, in the test tree only — it
is not production code. It speaks the canonical resolved tokens
(``constant`` / ``averaged`` / ``individual``); the retired truncated-basis
and legacy per-angle-agnostic tokens are not accepted.
"""

from __future__ import annotations

import numpy as np

#: The three in-scope canonical per-angle scaling modes the engine route covers.
IN_SCOPE_MODES = ("constant", "averaged", "individual")


def _validate(mode: str, n_physics: int, n_phi: int) -> None:
    if mode not in IN_SCOPE_MODES:
        raise ValueError(
            f"mode={mode!r} is not an in-scope canonical scaling mode; "
            f"expected one of {IN_SCOPE_MODES}."
        )
    if n_physics < 0:
        raise ValueError(f"n_physics must be non-negative, got {n_physics}")
    if n_phi < 1:
        raise ValueError(f"n_phi must be >= 1, got {n_phi}")


def _builder_head_len(mode: str, n_phi: int) -> int:
    """Length of the scaling head the pointwise builder emits for *mode*.

    ``constant`` -> 0 (frozen), ``averaged`` -> 2 (compressed scalars),
    ``individual`` -> ``2 * n_phi`` (per-angle contrast then offset).
    """
    if mode == "constant":
        return 0
    if mode == "averaged":
        return 2
    return 2 * n_phi  # individual


def expand_to_engine_scaling_first(
    vec: np.ndarray,
    *,
    n_physics: int,
    mode: str,
    n_phi: int,
) -> np.ndarray:
    """Expand the builder's canonical scaling-first ``p0`` to the engine's layout.

    Builder p0 (input):  ``[scaling_head | physics(n_physics)]`` where the head is
        empty (``constant``), ``[c_avg, o_avg]`` (``averaged``), or
        ``[contrast(n_phi) | offset(n_phi)]`` (``individual``).
    Engine vector (output): ``[contrast(n_phi) | offset(n_phi) | physics(n_physics)]``
        (or ``[physics]`` only for ``constant``).

    The conversion is the IDENTITY for ``constant`` and ``individual`` (the builder
    head already equals the engine layout); for ``averaged`` the 2 compressed
    scalars are broadcast to a ``2 * n_phi`` per-angle block (the engine has no
    compressed averaged mode).
    """
    _validate(mode, n_physics, n_phi)
    vec = np.asarray(vec, dtype=np.float64)
    head_len = _builder_head_len(mode, n_phi)
    expected_in = head_len + n_physics
    if vec.shape != (expected_in,):
        raise ValueError(
            f"scaling-first vec has shape {vec.shape}; expected ({expected_in},) "
            f"for mode={mode!r}, n_physics={n_physics}, n_phi={n_phi}."
        )

    if mode == "constant":
        # Physics-only; identity.
        return vec.copy()

    if mode == "averaged":
        contrast_scalar = vec[0]
        offset_scalar = vec[1]
        physics = vec[2:]
        contrast = np.full(n_phi, contrast_scalar, dtype=np.float64)
        offset = np.full(n_phi, offset_scalar, dtype=np.float64)
        return np.concatenate([contrast, offset, physics])

    # individual: builder head is already [contrast(n_phi) | offset(n_phi)] then
    # physics -> identical to the engine scaling-first layout.
    return vec.copy()
