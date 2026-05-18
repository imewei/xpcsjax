"""Config-driven dispatch returns the right physics model class.

Task 28: validates that ConfigManager (or the top-level ``make_model`` factory)
routes ``analysis_mode: two_component`` to :class:`HeterodyneModel` and the
homodyne modes (``static`` / ``laminar_flow``) to the legacy CombinedModel
path.
"""
from __future__ import annotations

import pytest


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


def _dispatch(cfg):
    """Call whichever dispatch surface the engine exposes (Task 28).

    Tries, in order:
      1. ``ConfigManager.get_model()`` / ``create_model()`` / ``build_model()``
      2. ``xpcsjax.core.models.make_model(cfg)``
    """
    for attr in ("get_model", "create_model", "build_model"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)()
    try:
        from xpcsjax.core.models import make_model
    except ImportError:
        pytest.skip("no dispatch surface found (ConfigManager.* or make_model)")
    return make_model(cfg)


def test_two_component_dispatches_to_heterodyne(tmp_path):
    """analysis_mode: two_component must produce a HeterodyneModel instance."""
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model import HeterodyneModel
    from xpcsjax.core.models import PhysicsModelBase

    config_path = _write_config(tmp_path, "two_component")
    cfg = ConfigManager(config_path)

    model = _dispatch(cfg)

    assert isinstance(model, HeterodyneModel), (
        f"expected HeterodyneModel for analysis_mode='two_component', "
        f"got {type(model).__name__}"
    )
    assert isinstance(model, PhysicsModelBase)


def test_heterodyne_synonym_dispatches_to_heterodyne(tmp_path):
    """analysis_mode: heterodyne (synonym) must also produce HeterodyneModel."""
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model import HeterodyneModel

    config_path = _write_config(tmp_path, "heterodyne")
    cfg = ConfigManager(config_path)

    model = _dispatch(cfg)

    assert isinstance(model, HeterodyneModel), (
        f"expected HeterodyneModel for analysis_mode='heterodyne' synonym, "
        f"got {type(model).__name__}"
    )


def test_static_does_not_dispatch_to_heterodyne(tmp_path):
    """analysis_mode: static must NOT produce a HeterodyneModel (sanity)."""
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model import HeterodyneModel

    config_path = _write_config(tmp_path, "static")
    cfg = ConfigManager(config_path)

    model = _dispatch(cfg)

    assert not isinstance(model, HeterodyneModel), (
        f"expected non-heterodyne model for analysis_mode='static', "
        f"got {type(model).__name__}"
    )


def test_laminar_flow_does_not_dispatch_to_heterodyne(tmp_path):
    """analysis_mode: laminar_flow must NOT produce a HeterodyneModel."""
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model import HeterodyneModel

    config_path = _write_config(tmp_path, "laminar_flow")
    cfg = ConfigManager(config_path)

    model = _dispatch(cfg)

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

    model = make_model({"analysis_mode": "static"})
    assert not isinstance(model, HeterodyneModel)


def test_config_manager_normalizes_heterodyne_synonym(tmp_path):
    """ConfigManager should normalize 'heterodyne' / 'Heterodyne' → 'two_component'."""
    from xpcsjax.config import ConfigManager

    config_path = _write_config(tmp_path, "Heterodyne")
    cfg = ConfigManager(config_path)

    assert cfg.config["analysis_mode"] == "two_component"
