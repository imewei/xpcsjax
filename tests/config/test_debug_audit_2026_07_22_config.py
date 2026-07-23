"""Regression tests for the 2026-07-22 config-module debug audit.

Five adversarially-verified defects across ``xpcsjax/config/``:

1. :meth:`ConfigManager.load_config` / :meth:`_validate_config_version` did
   ``self.config["metadata"].get(...)`` unguarded — a blank top-level
   ``metadata:`` YAML key parses to ``None`` and raised an ``AttributeError``
   (outside the ``load_config`` except tuple), crashing construction instead of
   falling back to defaults.
2. ``_KNOWN_TOP_LEVEL_KEYS`` was missing the heterodyne grouped-format keys
   (``parameters`` / ``temporal`` / ``scattering`` / ``scaling``), producing a
   false "Unknown top-level config keys" warning on valid two_component configs.
3. :meth:`ConfigManager.get_initial_parameters` injected per-angle
   contrast/offset keys AFTER the active/fixed filters, so a contrast/offset
   name listed in ``fixed_parameters`` leaked into the result.
4. :meth:`ParameterSpace.from_config` rejected the bare ``'static'`` alias that
   :class:`ConfigManager` accepts (via ``allow_bare_static=True``).
5. :meth:`ParameterManager._load_config_bounds` registered a new-parameter
   bounds entry missing ``min``/``max`` verbatim, deferring a confusing raw
   ``KeyError('min')`` to a much later ``get_bounds_as_arrays`` call.
"""

from __future__ import annotations

import pytest

from xpcsjax.config.manager import ConfigManager
from xpcsjax.config.parameter_manager import ParameterManager
from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.config.parameter_space import ParameterSpace


# --------------------------------------------------------------------------- #
# #1 — null top-level ``metadata`` must not crash construction
# --------------------------------------------------------------------------- #
def test_metadata_none_config_override_constructs() -> None:
    # Exercises _validate_config_version (site B, via _normalize_schema).
    mgr = ConfigManager(config_override={"metadata": None, "analysis_mode": "laminar_flow"})
    assert mgr.config["analysis_mode"] == "laminar_flow"


def test_metadata_none_yaml_file_constructs(tmp_path) -> None:
    # Exercises load_config (site A). Blank ``metadata:`` parses to None.
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("metadata:\nanalysis_mode: laminar_flow\n")
    mgr = ConfigManager(config_file=str(cfg))
    assert mgr.config["analysis_mode"] == "laminar_flow"


# --------------------------------------------------------------------------- #
# #2 — heterodyne grouped-format keys are known, not "typos"
# --------------------------------------------------------------------------- #
def test_grouped_format_keys_no_unknown_warning(caplog) -> None:
    config = {
        "analysis_mode": "two_component",
        "parameters": {},
        "temporal": {},
        "scattering": {},
        "scaling": {},
    }
    with caplog.at_level("WARNING"):
        ConfigManager(config_override=config)
    assert not any("Unknown top-level config keys" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# #3 — per-angle scaling keys obey fixed_parameters exclusion
# --------------------------------------------------------------------------- #
def test_fixed_per_angle_scaling_name_is_excluded() -> None:
    config = {
        "analysis_mode": "laminar_flow",
        "initial_parameters": {
            "parameter_names": ["D0", "alpha"],
            "values": [100.0, 1.0],
            "per_angle_scaling": {"contrast": [0.3, 0.4], "offset": [1.0, 1.1]},
            "fixed_parameters": {"contrast_0": 0.0},
        },
    }
    params = ConfigManager(config_override=config).get_initial_parameters()
    assert "contrast_0" not in params, "fixed per-angle scaling name must be excluded"
    assert "contrast_1" in params, "non-fixed injected key must be retained"
    assert {"D0", "alpha"} <= set(params)


# --------------------------------------------------------------------------- #
# #4 — ParameterSpace.from_config accepts bare 'static' like ConfigManager
# --------------------------------------------------------------------------- #
def test_parameter_space_accepts_bare_static() -> None:
    ps = ParameterSpace.from_config({"analysis_mode": "static"})
    assert ps.model_type == AnalysisMode.STATIC_ANISOTROPIC.value


# --------------------------------------------------------------------------- #
# #5 — new-parameter bounds without min/max raise a clear ValueError
# --------------------------------------------------------------------------- #
def test_new_param_bounds_missing_min_max_raises() -> None:
    config = {
        "analysis_mode": "laminar_flow",
        "parameter_space": {"bounds": [{"name": "my_new_param"}]},
    }
    with pytest.raises(ValueError, match=r"my_new_param.*min.*max|missing required key"):
        ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
