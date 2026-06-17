"""Regression tests for the 2026-06-17 debug-audit fixes (config layer)."""

from __future__ import annotations

import logging


# ---------------------------------------------------------------------------
# ParameterManager.get_parameter_bounds — cache hits must return separate
# BoundDict objects, not aliases into the cache (finding #26).
# ---------------------------------------------------------------------------
def test_bounds_cache_hit_returns_isolated_dicts() -> None:
    from xpcsjax.config.parameter_manager import ParameterManager

    pm = ParameterManager()
    first = pm.get_parameter_bounds(["D0"])  # cold path (populates cache)
    second = pm.get_parameter_bounds(["D0"])  # cache hit

    # Mutate the cache-hit result; a later call must be unaffected.
    second[0]["min"] = -999.0
    third = pm.get_parameter_bounds(["D0"])
    assert third[0]["min"] != -999.0
    # And the cold-path result must also be isolated from the hit result.
    assert first[0] is not second[0]


# ---------------------------------------------------------------------------
# ParameterManager.__repr__ — optimizable count must match the set-difference,
# never a cardinality subtraction that can go negative (finding #27).
# ---------------------------------------------------------------------------
def test_repr_optimizable_matches_set_difference() -> None:
    from xpcsjax.config.parameter_manager import ParameterManager

    # fixed_parameters carrying scaling/foreign names that are NOT in the
    # active (physics) set would make len(active) - len(fixed) wrong/negative.
    config = {
        "initial_parameters": {
            "active_parameters": ["D0", "alpha"],
            "fixed_parameters": {
                "contrast": 0.5,
                "offset": 1.0,
                "not_a_real_param": 0.0,
            },
        },
    }
    pm = ParameterManager(config)
    expected = len(pm.get_optimizable_parameters())
    text = repr(pm)
    assert f"optimizable={expected}" in text
    assert "optimizable=-" not in text  # never negative


# ---------------------------------------------------------------------------
# ConfigManager._log_key_config_values — a non-numeric memory_fraction must
# warn, not raise a TypeError (finding #18).
# ---------------------------------------------------------------------------
def test_memory_fraction_string_does_not_crash(caplog) -> None:
    from xpcsjax.config.manager import ConfigManager

    cm = ConfigManager.__new__(ConfigManager)
    cm.config = {
        "optimization": {"method": "nlsq", "nlsq": {"memory_fraction": "0.5"}},
        "experimental_data": {},
    }
    with caplog.at_level(logging.WARNING):
        cm._log_key_config_values()  # must not raise
    assert any("memory_fraction" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# ConfigManager._normalize_experimental_data — a present-but-null phi path must
# not raise, and a None data-folder must not suppress phi normalization
# (findings #17 + uncertain #4).
# ---------------------------------------------------------------------------
def test_phi_angles_null_path_does_not_crash() -> None:
    from xpcsjax.config.manager import ConfigManager

    cm = ConfigManager.__new__(ConfigManager)
    cm.config = {
        "experimental_data": {
            "phi_angles_path": None,
            "phi_angles_file": None,
        },
    }
    cm._normalize_experimental_data()  # must not raise


def test_phi_normalization_runs_when_data_folder_is_none() -> None:
    from xpcsjax.config.manager import ConfigManager

    cm = ConfigManager.__new__(ConfigManager)
    cm.config = {
        "experimental_data": {
            "data_folder_path": None,
            "data_file_name": None,
            "phi_angles_path": "/tmp/angles",
            "phi_angles_file": "phi.txt",
        },
    }
    cm._normalize_experimental_data()
    # phi composite path is still produced despite the None data-folder fields.
    assert cm.config["experimental_data"]["phi_angles_full_path"].endswith("phi.txt")
