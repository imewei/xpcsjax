"""Single source of truth for the resolved per-angle scaling mode (spec §4 Seam 1).

Collapses the per-angle scaling contract to exactly three RESOLVED variants —
``constant`` (frozen), ``averaged`` (one optimized pair), ``individual`` (per-angle
optimized) — and provides the sole owner of the ``constant_scaling_threshold``
default. ``auto`` is input sugar resolving to ``averaged``/``individual``. The
removed ``fourier`` and legacy ``independent`` tokens are rejected by the generic
``else`` branch (no special-case arm).

Phase 0: pure unit; no call site imports this yet.
"""

from __future__ import annotations

from typing import Literal, get_args

PerAngleMode = Literal["constant", "averaged", "individual"]

DEFAULT_CONSTANT_SCALING_THRESHOLD = 3
"""Sole owner of the auto->averaged/individual cutover default (spec §4 Seam 1)."""

_RESOLVED: frozenset[str] = frozenset(get_args(PerAngleMode))


def resolve_per_angle_mode(
    token: str,
    n_phi: int,
    constant_scaling_threshold: int = DEFAULT_CONSTANT_SCALING_THRESHOLD,
) -> PerAngleMode:
    """Resolve a user/config ``per_angle_mode`` token to a canonical variant.

    Parameters
    ----------
    token : str
        One of ``"constant"``, ``"averaged"``, ``"individual"``, or ``"auto"``.
    n_phi : int
        Number of unique phi angles (only consulted for ``"auto"``).
    constant_scaling_threshold : int, optional
        ``auto`` resolves to ``"averaged"`` when ``n_phi >= threshold`` else
        ``"individual"``. Defaults to :data:`DEFAULT_CONSTANT_SCALING_THRESHOLD`.

    Returns
    -------
    PerAngleMode
        The resolved variant: ``"constant"``, ``"averaged"``, or ``"individual"``.

    Raises
    ------
    ValueError
        For any token other than the four accepted strings — including the
        removed ``"fourier"`` and the legacy ``"independent"`` alias.
    """
    if token in _RESOLVED:
        return token  # type: ignore[return-value]
    if token == "auto":
        threshold = max(int(constant_scaling_threshold), 1)
        return "averaged" if n_phi >= threshold else "individual"
    raise ValueError(
        f"unknown per_angle_mode {token!r}; valid: "
        "constant, averaged, individual, auto"
    )


def n_optimized(mode: PerAngleMode, n_phi: int) -> int:
    """Number of OPTIMIZED scaling parameters for a resolved mode.

    ``constant`` -> 0 (frozen), ``averaged`` -> 2, ``individual`` -> ``2 * n_phi``.

    Raises
    ------
    ValueError
        If ``mode`` is not already a resolved variant (e.g. ``"auto"`` or a
        removed token); callers must resolve first via
        :func:`resolve_per_angle_mode`.
    """
    if mode == "constant":
        return 0
    if mode == "averaged":
        return 2
    if mode == "individual":
        return 2 * int(n_phi)
    raise ValueError(
        f"unknown per_angle_mode {mode!r}; valid: constant, averaged, individual"
    )
