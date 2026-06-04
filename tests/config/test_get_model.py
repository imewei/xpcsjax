"""Config-driven dispatch returns the right physics model class.

Task 28: validates that :meth:`ConfigManager.get_model` routes
``analysis_mode: two_component`` to :class:`HeterodyneModel` and the homodyne
modes (``static`` / ``laminar_flow``) to the legacy ``CombinedModel`` path.
"""

from __future__ import annotations


def _write_config(tmp_path, mode: str) -> str:
    """Minimal YAML config setting analysis_mode."""
    cfg = tmp_path / f"{mode}.yaml"
    cfg.write_text(
        f"""
analysis_mode: "{mode}"
analyzer_parameters:
  dt: 1.0
  start_frame: 1
  end_frame: 10
  scattering:
    wavevector_q: 0.01
experimental_data:
  data_folder_path: "/tmp"
  data_file_name: "dummy.hdf"
  phi_angles_path: "/tmp"
  phi_angles_file: "phi.txt"
"""
    )
    return str(cfg)


def test_two_component_dispatches_to_heterodyne(tmp_path):
    """analysis_mode: two_component must produce a HeterodyneModel instance."""
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model import HeterodyneModel
    from xpcsjax.core.models import PhysicsModelBase

    cfg = ConfigManager(_write_config(tmp_path, "two_component"))
    model = cfg.get_model()

    assert isinstance(model, HeterodyneModel), (
        f"expected HeterodyneModel for analysis_mode='two_component', got {type(model).__name__}"
    )
    assert isinstance(model, PhysicsModelBase)


def test_heterodyne_synonym_dispatches_to_heterodyne(tmp_path):
    """analysis_mode: heterodyne (synonym) must also produce HeterodyneModel."""
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model import HeterodyneModel

    cfg = ConfigManager(_write_config(tmp_path, "heterodyne"))
    model = cfg.get_model()

    assert isinstance(model, HeterodyneModel), (
        f"expected HeterodyneModel for analysis_mode='heterodyne' synonym, "
        f"got {type(model).__name__}"
    )


def test_static_does_not_dispatch_to_heterodyne(tmp_path):
    """analysis_mode: static_anisotropic must NOT produce a HeterodyneModel (sanity)."""
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model import HeterodyneModel

    cfg = ConfigManager(_write_config(tmp_path, "static_anisotropic"))
    model = cfg.get_model()

    assert not isinstance(model, HeterodyneModel), (
        f"expected non-heterodyne model for analysis_mode='static_anisotropic', "
        f"got {type(model).__name__}"
    )


def test_laminar_flow_does_not_dispatch_to_heterodyne(tmp_path):
    """analysis_mode: laminar_flow must NOT produce a HeterodyneModel."""
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model import HeterodyneModel

    cfg = ConfigManager(_write_config(tmp_path, "laminar_flow"))
    model = cfg.get_model()

    assert not isinstance(model, HeterodyneModel), (
        f"expected non-heterodyne model for analysis_mode='laminar_flow', "
        f"got {type(model).__name__}"
    )


def test_make_model_accepts_dict():
    """make_model should also work on a raw config dict (no ConfigManager)."""
    from xpcsjax.core.heterodyne_model import HeterodyneModel
    from xpcsjax.core.models import make_model

    model = make_model({"analysis_mode": "two_component"})
    assert isinstance(model, HeterodyneModel)

    model = make_model({"analysis_mode": "static_anisotropic"})
    assert not isinstance(model, HeterodyneModel)


def test_config_manager_normalizes_heterodyne_synonym(tmp_path):
    """ConfigManager should normalize 'heterodyne' / 'Heterodyne' → 'two_component'."""
    from xpcsjax.config import ConfigManager

    cfg = ConfigManager(_write_config(tmp_path, "Heterodyne"))
    assert cfg.config["analysis_mode"] == "two_component"
