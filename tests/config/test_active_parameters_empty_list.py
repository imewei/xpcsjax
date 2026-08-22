"""Regression: initial_parameters.active_parameters: [] must mean 'fix everything',
not 'absent -> fall back to mode defaults'."""

from xpcsjax.config.parameter_manager import ParameterManager
from xpcsjax.config.parameter_registry import AnalysisMode


def test_empty_active_parameters_list_means_none_active():
    config = {"analysis_mode": "laminar_flow", "initial_parameters": {"active_parameters": []}}
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    assert pm.get_active_parameters() == []


def test_missing_active_parameters_key_falls_back_to_defaults():
    config = {"analysis_mode": "laminar_flow", "initial_parameters": {}}
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    assert len(pm.get_active_parameters()) == 7  # unchanged: absent key still uses mode defaults
