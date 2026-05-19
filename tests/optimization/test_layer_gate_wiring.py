"""Task 29 follow-up: verify the model-lineage gate is wired through the
production construction path (``AntiDegeneracyController.from_config``), not
only via the direct constructor.

Codex flagged in 2026-05-19 review that ``from_config`` accepted
``is_laminar_flow`` but never ``analysis_mode``, so the ``_LAYER_GATES``
defense-in-depth was dormant: production heterodyne fits relied solely on the
upstream ``is_laminar_flow`` guard at both call sites (``core.py`` and
``strategies/stratified_ls.py``) to suppress Layer 5. If that guard is ever
relaxed (e.g. to let Layers 1-4 fire for heterodyne), Layer 5 would silently
fire too without this wiring.
"""
from __future__ import annotations

import numpy as np

from xpcsjax.optimization.nlsq.anti_degeneracy_controller import (
    AntiDegeneracyController,
)

_MIN_CONFIG = {
    "enable": True,
    "per_angle_mode": "individual",
    "shear_weighting": {"enable": True},
}


def _build(analysis_mode: str | None) -> AntiDegeneracyController:
    """Construct a controller through the production API."""
    return AntiDegeneracyController.from_config(
        config_dict=_MIN_CONFIG,
        n_phi=3,
        phi_angles=np.deg2rad(np.array([0.0, 60.0, 120.0])),
        n_physical=7,
        per_angle_scaling=True,
        is_laminar_flow=True,
        analysis_mode=analysis_mode,
    )


def test_from_config_accepts_analysis_mode_kwarg():
    """The production constructor must accept ``analysis_mode`` so callers can
    thread model lineage through. Without it the lineage gate is unreachable
    via the production API."""
    controller = _build(analysis_mode="two_component")
    assert controller.analysis_mode == "two_component"


def test_layer5_gated_off_for_two_component():
    """Lineage gate must short-circuit ShearSensitivityWeighting for heterodyne
    (two_component) fits, independent of whatever upstream ``is_laminar_flow``
    flag the call site happened to pass."""
    controller = _build(analysis_mode="two_component")
    assert controller.is_layer_active("ShearSensitivityWeighting") is False


def test_layer5_active_for_laminar_flow():
    """Homodyne laminar_flow path must keep Layer 5 active — this is the
    regime the homodyne characterization gate (rtol=1e-10) certifies."""
    controller = _build(analysis_mode="laminar_flow")
    assert controller.is_layer_active("ShearSensitivityWeighting") is True


def test_layer5_active_when_lineage_omitted():
    """Backward-compat: omitting ``analysis_mode`` keeps existing behavior
    (all layers active). The homodyne characterization tests rely on this."""
    controller = _build(analysis_mode=None)
    assert controller.is_layer_active("ShearSensitivityWeighting") is True
