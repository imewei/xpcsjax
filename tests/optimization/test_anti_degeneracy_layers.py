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


# ---------------------------------------------------------------------------
# Task 5: execute_layers flag registration (INERT gate, default False)
# ---------------------------------------------------------------------------


class TestExecuteLayersFlag:
    """``execute_layers`` is a registered, parseable, INERT config gate.

    All tests assert the flag parses correctly and that no behavioral change
    occurs (controller activation markers are identical whether the flag is
    absent, False, or True).
    """

    def test_config_default_is_false_when_key_absent(self) -> None:
        """``from_dict({})`` must give ``execute_layers == False``."""
        from xpcsjax.optimization.nlsq.anti_degeneracy_controller import (
            AntiDegeneracyConfig,
        )

        cfg = AntiDegeneracyConfig.from_dict({})
        assert cfg.execute_layers is False

    def test_config_parses_false_explicitly(self) -> None:
        """``from_dict({"execute_layers": False})`` must give ``False``."""
        from xpcsjax.optimization.nlsq.anti_degeneracy_controller import (
            AntiDegeneracyConfig,
        )

        cfg = AntiDegeneracyConfig.from_dict({"execute_layers": False})
        assert cfg.execute_layers is False

    def test_config_parses_true_explicitly(self) -> None:
        """``from_dict({"execute_layers": True})`` must give ``True``."""
        from xpcsjax.optimization.nlsq.anti_degeneracy_controller import (
            AntiDegeneracyConfig,
        )

        cfg = AntiDegeneracyConfig.from_dict({"execute_layers": True})
        assert cfg.execute_layers is True

    def test_dataclass_default_field_is_false(self) -> None:
        """The dataclass default must be ``False`` without calling ``from_dict``."""
        from xpcsjax.optimization.nlsq.anti_degeneracy_controller import (
            AntiDegeneracyConfig,
        )

        cfg = AntiDegeneracyConfig()
        assert cfg.execute_layers is False

    def test_controller_property_exposes_config_value(self) -> None:
        """``controller.execute_layers`` must proxy ``controller.config.execute_layers``."""
        import numpy as np

        from xpcsjax.optimization.nlsq.anti_degeneracy_controller import (
            AntiDegeneracyController,
        )

        phi_angles = np.array([0.0, 30.0, 60.0])
        controller = AntiDegeneracyController.from_config(
            config_dict={},
            n_phi=3,
            phi_angles=phi_angles,
            n_physical=7,
        )
        assert controller.execute_layers is False
        assert controller.execute_layers == controller.config.execute_layers

    def test_controller_execute_layers_true_does_not_change_activation_markers(
        self,
    ) -> None:
        """Enabling ``execute_layers`` must NOT alter any activation marker.

        With the flag absent (False) vs True, ``get_diagnostics()`` must be
        identical except for the ``execute_layers`` key itself.  All
        hierarchical_active/regularization_active markers must stay unchanged,
        confirming the flag is currently INERT.
        """
        import numpy as np

        from xpcsjax.optimization.nlsq.anti_degeneracy_controller import (
            AntiDegeneracyController,
        )

        phi_angles = np.linspace(0.0, 3.0, 5)

        ctrl_off = AntiDegeneracyController.from_config(
            config_dict={"execute_layers": False},
            n_phi=5,
            phi_angles=phi_angles,
            n_physical=7,
        )
        ctrl_on = AntiDegeneracyController.from_config(
            config_dict={"execute_layers": True},
            n_phi=5,
            phi_angles=phi_angles,
            n_physical=7,
        )

        diag_off = ctrl_off.get_diagnostics()
        diag_on = ctrl_on.get_diagnostics()

        # execute_layers itself must differ
        assert diag_off["execute_layers"] is False
        assert diag_on["execute_layers"] is True

        # Key SETS must match — a future bug that adds a conditional key is caught here
        assert set(diag_off.keys()) == set(diag_on.keys()), (
            f"key sets differ: {set(diag_off) ^ set(diag_on)}"
        )

        # All other keys must be identical — the flag is INERT
        keys_to_compare = [k for k in diag_off if k != "execute_layers"]
        for key in keys_to_compare:
            assert diag_off[key] == diag_on[key], (
                f"get_diagnostics()[{key!r}] differs when execute_layers changes: "
                f"{diag_off[key]!r} vs {diag_on[key]!r}. "
                f"The flag must be inert."
            )

    def test_templates_contain_execute_layers_false(self) -> None:
        """All FOUR YAML templates must parse and expose ``execute_layers: false``."""
        import pathlib

        import yaml

        templates_dir = (
            pathlib.Path(__file__).parents[2]
            / "xpcsjax"
            / "config"
            / "templates"
        )
        for template_name in (
            "xpcsjax_laminar_flow.yaml",
            "xpcsjax_two_component.yaml",
            "xpcsjax_static_anisotropic.yaml",
            "xpcsjax_static_isotropic.yaml",
        ):
            path = templates_dir / template_name
            assert path.exists(), f"Template not found: {path}"
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            ad_block = (
                data.get("optimization", {}).get("nlsq", {}).get("anti_degeneracy", {})
            )
            assert "execute_layers" in ad_block, (
                f"{template_name}: 'execute_layers' key missing from anti_degeneracy block"
            )
            assert ad_block["execute_layers"] is False, (
                f"{template_name}: execute_layers must default to false, "
                f"got {ad_block['execute_layers']!r}"
            )


class TestExecuteLayersNLSQConfigHomodyne:
    """``execute_layers`` round-trips through the homodyne solver ``NLSQConfig``.

    The flag is registered for round-trip completeness only — it is INERT
    (nothing reads it to branch behavior). ``to_dict`` emits it as a TOP-LEVEL
    key inside the nested ``anti_degeneracy`` block, mirroring ``per_angle_mode``.
    """

    def test_from_dict_default_is_false_when_key_absent(self) -> None:
        """``from_dict({})`` must give ``execute_layers is False``."""
        from xpcsjax.optimization.nlsq.config import NLSQConfig

        cfg = NLSQConfig.from_dict({})
        assert cfg.execute_layers is False

    def test_from_dict_parses_true_from_nested_block(self) -> None:
        """A nested ``anti_degeneracy.execute_layers`` value must be parsed."""
        from xpcsjax.optimization.nlsq.config import NLSQConfig

        cfg = NLSQConfig.from_dict({"anti_degeneracy": {"execute_layers": True}})
        assert cfg.execute_layers is True

    def test_to_dict_emits_nested_execute_layers(self) -> None:
        """``to_dict()["anti_degeneracy"]["execute_layers"]`` echoes the field."""
        from xpcsjax.optimization.nlsq.config import NLSQConfig

        for value in (True, False):
            cfg = NLSQConfig(execute_layers=value)
            assert cfg.to_dict()["anti_degeneracy"]["execute_layers"] is value

    def test_to_dict_from_dict_roundtrip(self) -> None:
        """``from_dict(to_dict())`` preserves ``execute_layers`` both ways."""
        from xpcsjax.optimization.nlsq.config import NLSQConfig

        for value in (True, False):
            cfg = NLSQConfig(execute_layers=value)
            restored = NLSQConfig.from_dict(cfg.to_dict())
            assert restored.execute_layers is value


class TestExecuteLayersNLSQConfigHeterodyne:
    """``execute_layers`` round-trips through the heterodyne solver ``NLSQConfig``.

    The flag is registered for round-trip completeness only — it is INERT
    (nothing reads it to branch behavior). The heterodyne ``to_dict`` emits it
    as a FLAT top-level key, mirroring ``enable_hierarchical``.
    """

    def test_from_dict_default_is_false_when_key_absent(self) -> None:
        """``from_dict({})`` must give ``execute_layers is False``."""
        from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig

        cfg = NLSQConfig.from_dict({})
        assert cfg.execute_layers is False

    def test_from_dict_parses_true_from_nested_block(self) -> None:
        """A nested ``anti_degeneracy.execute_layers`` value must be parsed."""
        from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig

        cfg = NLSQConfig.from_dict({"anti_degeneracy": {"execute_layers": True}})
        assert cfg.execute_layers is True

    def test_to_dict_emits_flat_execute_layers(self) -> None:
        """``to_dict()["execute_layers"]`` echoes the field (flat key)."""
        from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig

        for value in (True, False):
            cfg = NLSQConfig(execute_layers=value)
            assert cfg.to_dict()["execute_layers"] is value

    def test_to_dict_from_dict_roundtrip(self) -> None:
        """``from_dict(to_dict())`` preserves ``execute_layers`` both ways."""
        from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig

        for value in (True, False):
            cfg = NLSQConfig(execute_layers=value)
            restored = NLSQConfig.from_dict(cfg.to_dict())
            assert restored.execute_layers is value
