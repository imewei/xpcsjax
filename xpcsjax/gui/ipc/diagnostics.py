"""Map engine diagnostics + log banners into GUI live-diagnostics events.

JAX-free: stdlib + the event schema only. Used by the worker (to emit
``LayerStatus`` post-fit and ``Banner`` from recognized log lines) and importable
by the GUI process.
"""

from __future__ import annotations

from xpcsjax.service.events import Banner, BannerKind

# L5 (shear weighting) reports a sentinel string when inactive; treat any value
# containing these markers (or falsy) as "not active".
_L5_INACTIVE_MARKERS = ("inactive", "not_applicable")


def _l5_active(value: object) -> bool:
    if not value:
        return False
    text = str(value).lower()
    return not any(marker in text for marker in _L5_INACTIVE_MARKERS)


def layer_status_from_diagnostics(diagnostics: dict | None) -> dict[str, bool]:
    """Map ``result.nlsq_diagnostics`` to an L1–L5 active map.

    L1 (per-angle reparameterization) is always active. L2/L3/L4 read the
    ``hierarchical_active`` / ``regularization_active`` / ``gradient_monitor``
    keys; L5 reads ``shear_weighting`` (sentinels -> inactive).
    """
    diag = diagnostics or {}
    return {
        "L1": True,
        "L2": bool(diag.get("hierarchical_active", False)),
        "L3": bool(diag.get("regularization_active", False)),
        "L4": bool(diag.get("gradient_monitor")),
        "L5": _l5_active(diag.get("shear_weighting")),
    }


def classify_banner(level: str, message: str) -> Banner | None:
    """Classify an engine log line into a Banner, or None if it is not a banner.

    Recognizes the stable engine prefixes; ``run_id``/``seq`` are placeholders
    restamped by :class:`~xpcsjax.gui.ipc.emitter.EventEmitter`.
    """
    text = message.strip()
    # Engine emits two anti-degeneracy banner families: the controller's
    # "ANTI-DEGENERACY: Layer N ..." and the streaming/heterodyne
    # "ANTI-DEGENERACY DEFENSE: Layer N ..." / "ANTI-DEGENERACY DEFENSE [AS EXECUTED] ...".
    if text.startswith("ANTI-DEGENERACY") and ("Layer" in text or "[AS EXECUTED]" in text):
        kind = BannerKind.INFO
    elif "GRADIENT COLLAPSE" in text:
        kind = BannerKind.GRADIENT_COLLAPSE
    elif text.startswith("[CMA-ES]"):
        kind = BannerKind.CMAES_ESCAPE
    else:
        return None
    return Banner(run_id="", seq=0, text=text, kind=kind)
