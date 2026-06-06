"""Shared per-angle-mode banner emitter (laminar ↔ heterodyne parity).

Pure logging: emits log records only; touches no solver state and no numerics,
so it is invisible to the rtol=1e-10 parity gates. Each caller passes its OWN
module logger so the record's name matches the calling path while the banner
text stays byte-identical across modes.
"""

from __future__ import annotations

import logging
from typing import Any

_W60 = "=" * 60

# Resolved controller mode (``per_angle_mode_actual``) -> short banner token.
# Public: imported by hybrid_streaming (which has no controller object).
MODE_SHORT = {
    "auto_averaged": "averaged",
    "fixed_constant": "constant",
    "individual": "individual",
    "fourier": "fourier",
}


def log_effective_per_angle_mode(
    logger: logging.Logger | logging.LoggerAdapter[Any],
    *,
    mode: str,
    n_phi: int,
    n_physics: int,
    n_scaling: int,
    threshold: int | None = None,
) -> None:
    """Emit the unified 'Effective per-angle mode' banner under ``logger``.

    Parameters
    ----------
    logger : logging.Logger or logging.LoggerAdapter
        Caller's own logger; the emitted record's ``name`` matches the caller's
        module path so log routing is preserved.
    mode : str
        Short mode token: ``"averaged"``, ``"individual"``, ``"fourier"``, or
        ``"constant"``.
    n_phi : int
        Number of phi angles in the fit.
    n_physics : int
        Number of physical parameters (7 for laminar_flow, 14 for
        two_component).
    n_scaling : int
        Count of OPTIMIZED scaling parameters (0 for ``"constant"``).
    threshold : int or None
        ``constant_scaling_threshold`` from config. When provided the Reason
        line is emitted; when ``None`` the Reason line is skipped.
    """
    logger.info(_W60)
    logger.info("ANTI-DEGENERACY: Effective per-angle mode '%s'", mode)
    if threshold is not None:
        rel = ">=" if n_phi >= threshold else "<"
        logger.info(
            "  Reason: n_phi (%d) %s constant_scaling_threshold (%d)",
            n_phi,
            rel,
            threshold,
        )
    if mode == "constant":
        logger.info(
            "  Parameters: %d physical only (per-angle scaling fixed from quantiles)",
            n_physics,
        )
    else:
        if mode == "averaged":
            descr = f"{n_scaling} averaged scaling"
        elif mode == "fourier":
            descr = f"{n_scaling} Fourier coeffs"
        elif mode == "individual":
            descr = f"{n_scaling} per-angle scaling"
        else:
            descr = f"{n_scaling} scaling"
        logger.info(
            "  Parameters: %d physical + %s = %d total",
            n_physics,
            descr,
            n_physics + n_scaling,
        )
    logger.info(_W60)


def log_effective_mode_from_controller(
    logger: logging.Logger | logging.LoggerAdapter[Any], controller: Any
) -> None:
    """Emit the unified banner from a resolved ``AntiDegeneracyController``.

    Convenience wrapper for paths that hold a controller (stratified-LS,
    CMA-ES). Pulls the short mode, the real ``n_physical``, and the resolved
    optimized scaling count (``n_per_angle_params``, which already reflects the
    Fourier fallback to ``2*n_phi``). No-ops when the controller is not
    enabled, so a disabled/uninitialized controller emits nothing.

    Parameters
    ----------
    logger : logging.Logger or logging.LoggerAdapter
        Caller's own logger.
    controller : AntiDegeneracyController
        A fully initialized controller (``_is_initialized`` is ``True``).
    """
    if not getattr(controller, "is_enabled", False):
        return
    mode = MODE_SHORT.get(
        controller.per_angle_mode_actual, controller.per_angle_mode_actual
    )
    # Explicit 'constant' is not threshold-selected → no Reason line.
    threshold = None if mode == "constant" else controller.config.constant_scaling_threshold
    log_effective_per_angle_mode(
        logger,
        mode=mode,
        n_phi=controller.n_phi,
        n_physics=controller.n_physical,
        n_scaling=controller.n_per_angle_params,
        threshold=threshold,
    )
