"""Verify all 5 anti-degeneracy layers ported over from homodyne.

Task 29 tests the Layer-5 model-lineage gating. This test catches a different
regression: did the dead-code removal refactoring accidentally cut into the
anti-degeneracy controller's layer wiring?"""

import inspect
from typing import Any

from xpcsjax.optimization.nlsq.anti_degeneracy_controller import (
    AntiDegeneracyController,
)

LAYER_NAMES = (
    "FourierReparameterizer",
    "HierarchicalOptimizer",
    "AdaptiveRegularizer",
    "GradientCollapseMonitor",
    "ShearSensitivityWeighting",
)


def test_controller_source_references_all_5_layers():
    """Static check: the controller class source must mention every layer name.

    If a layer class was dropped during the verbatim port, this catches it
    without needing to instantiate or introspect the controller's runtime state."""
    src = inspect.getsource(AntiDegeneracyController)
    missing = [name for name in LAYER_NAMES if name not in src]
    assert not missing, (
        f"AntiDegeneracyController source missing references to: {missing}. "
        f"Likely cause: a layer was dropped during the verbatim port "
        f"or during dead-code removal refactoring."
    )


def test_module_exports_all_5_layer_classes():
    """All 5 layer class names must be importable from the controller module
    (either re-exported by the controller module itself, or by its sibling
    modules in xpcsjax.optimization.nlsq).

    Pyright stale-indexing can hide these — the test asserts true reachability."""
    import importlib

    candidate_modules = [
        "xpcsjax.optimization.nlsq.anti_degeneracy_controller",
        "xpcsjax.optimization.nlsq.fourier_reparam",
        "xpcsjax.optimization.nlsq.hierarchical",
        "xpcsjax.optimization.nlsq.adaptive_regularization",
        "xpcsjax.optimization.nlsq.gradient_monitor",
        "xpcsjax.optimization.nlsq.shear_weighting",
    ]
    found: dict[str, str] = {}
    for module_path in candidate_modules:
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            continue
        for layer_name in LAYER_NAMES:
            if hasattr(mod, layer_name) and layer_name not in found:
                found[layer_name] = module_path

    missing = [name for name in LAYER_NAMES if name not in found]
    assert not missing, (
        f"Layer classes not importable from any of the candidate modules: {missing}. Found: {found}"
    )


def test_controller_instantiates_with_minimal_config():
    """The controller must construct from a minimal (config, n_phi, n_physical, phi_angles) tuple.

    Signature inspection adapts the call to whatever the ported controller
    actually expects. Asserts the constructor runs without error."""
    import numpy as np

    sig = inspect.signature(AntiDegeneracyController.__init__)
    # All non-self params:
    params = [p for p in sig.parameters.values() if p.name != "self"]
    arg_names = [p.name for p in params if p.default is inspect.Parameter.empty]

    # Build a minimal stub for each required arg
    stub_values: dict[str, Any] = {}
    n_phi_test, n_physical_test = 3, 7
    phi_angles_test = np.array([0.0, 30.0, 60.0])
    for name in arg_names:
        if name == "config":
            stub_values[name] = {}  # empty dict; many homodyne paths accept this
        elif name == "n_phi":
            stub_values[name] = n_phi_test
        elif name == "n_physical":
            stub_values[name] = n_physical_test
        elif name == "phi_angles":
            stub_values[name] = phi_angles_test
        else:
            # Unknown required arg — fail loudly so the test is updated
            raise AssertionError(
                f"AntiDegeneracyController has an unrecognized required arg: {name!r}. "
                f"Update this test."
            )

    controller = AntiDegeneracyController(**stub_values)
    assert controller is not None


def test_anti_degeneracy_config_overrides_defaults():
    """Homodyne AntiDegeneracyConfig.from_dict must honor config-file values
    over the dataclass defaults (constant_scaling_threshold=3,
    fourier_auto_threshold=6) — never silently dropped to the default."""
    from xpcsjax.optimization.nlsq.anti_degeneracy_controller import (
        AntiDegeneracyConfig,
    )

    defaults = AntiDegeneracyConfig()
    cfg = AntiDegeneracyConfig.from_dict(
        {
            "per_angle_mode": "individual",
            "constant_scaling_threshold": 7,
            "fourier_auto_threshold": 11,
        }
    )
    assert cfg.per_angle_mode == "individual"
    assert cfg.constant_scaling_threshold == 7 != defaults.constant_scaling_threshold
    assert cfg.fourier_auto_threshold == 11 != defaults.fourier_auto_threshold
