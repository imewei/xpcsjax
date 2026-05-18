"""ShearSensitivityWeighting (anti-degeneracy Layer 5) is gated by model lineage.

Per spec §10.3:
- Homodyne modes (static / static_isotropic / laminar_flow) -> 5 layers active including Layer 5
- Heterodyne mode (two_component) -> 4 layers; Layer 5 disabled (no shear rate to weight)
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
        config={}, n_phi=n_phi, n_physical=n_physical, phi_angles=phi_angles,
        **kwargs,
    )


@pytest.mark.parametrize("homodyne_mode", ["static", "static_isotropic", "laminar_flow"])
def test_shear_layer_active_for_homodyne(homodyne_mode):
    """Layer 5 is active for every homodyne analysis_mode."""
    controller = _make_controller(analysis_mode=homodyne_mode)
    assert controller.is_layer_active("ShearSensitivityWeighting") is True, (
        f"Layer 5 must be active for homodyne mode {homodyne_mode!r}"
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
