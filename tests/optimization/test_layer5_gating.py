"""ShearSensitivityWeighting (anti-degeneracy Layer 5) is gated by analysis mode.

L5 up-weights data near the flow direction phi0 to exploit the shear-sensitivity
peak, which only exists when the kernel has a shear rate. So L5 is active for
``laminar_flow`` ONLY:
- ``laminar_flow`` -> Layer 5 active (has shear rate / flow direction)
- ``static_isotropic`` / ``static_anisotropic`` -> Layer 5 disabled (no flow, no shear peak)
- ``two_component`` (heterodyne) -> Layer 5 disabled (no shear rate to weight)
"""

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.anti_degeneracy_controller import (
    AntiDegeneracyController,
)


def _make_controller(analysis_mode: str | None = None):
    """Build a controller with minimal-but-valid arguments."""
    n_phi, n_physical = 3, 7
    phi_angles = np.array([0.0, 30.0, 60.0])
    kwargs = {}
    if analysis_mode is not None:
        kwargs["analysis_mode"] = analysis_mode
    return AntiDegeneracyController(
        config={},
        n_phi=n_phi,
        n_physical=n_physical,
        phi_angles=phi_angles,
        **kwargs,
    )


def test_shear_layer_active_for_laminar_flow():
    """Layer 5 is active ONLY for laminar_flow (the mode with a shear rate)."""
    controller = _make_controller(analysis_mode="laminar_flow")
    assert controller.is_layer_active("ShearSensitivityWeighting") is True, (
        "Layer 5 must be active for laminar_flow (has flow direction / shear peak)"
    )


@pytest.mark.parametrize("static_mode", ["static_anisotropic", "static_isotropic"])
def test_shear_layer_inactive_for_static(static_mode):
    """Layer 5 is inactive for static modes — no flow direction, no shear peak."""
    controller = _make_controller(analysis_mode=static_mode)
    assert controller.is_layer_active("ShearSensitivityWeighting") is False, (
        f"Layer 5 must be inactive for static mode {static_mode!r} (no shear to weight)"
    )


def test_shear_layer_inactive_for_heterodyne():
    """Layer 5 is inactive for two_component (heterodyne) mode."""
    controller = _make_controller(analysis_mode="two_component")
    assert controller.is_layer_active("ShearSensitivityWeighting") is False, (
        "Layer 5 must be inactive for two_component mode (no shear rate to weight)"
    )


def test_heterodyne_synonym_also_inactive():
    """The 'heterodyne' synonym must produce the same gating."""
    controller = _make_controller(analysis_mode="heterodyne")
    assert controller.is_layer_active("ShearSensitivityWeighting") is False


def test_other_four_layers_active_for_heterodyne():
    """The other 4 anti-degeneracy layers stay active for heterodyne fits."""
    controller = _make_controller(analysis_mode="two_component")
    for name in (
        "FourierReparameterizer",
        "HierarchicalOptimizer",
        "AdaptiveRegularizer",
        "GradientCollapseMonitor",
    ):
        assert controller.is_layer_active(name) is True, (
            f"Layer {name!r} should remain active for heterodyne fits"
        )


def test_default_no_mode_keeps_all_layers_active():
    """Backward-compat: passing no analysis_mode -> all layers active (original behavior).

    This preserves the homodyne characterization gate's rtol=1e-10 behavior."""
    controller = _make_controller(analysis_mode=None)
    for layer in (
        "FourierReparameterizer",
        "HierarchicalOptimizer",
        "AdaptiveRegularizer",
        "GradientCollapseMonitor",
        "ShearSensitivityWeighting",
    ):
        assert controller.is_layer_active(layer) is True, (
            f"Layer {layer!r} should be active by default (no mode specified)"
        )
