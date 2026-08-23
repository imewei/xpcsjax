"""tied_parameters is implemented only for two_component (see
xpcsjax/config/heterodyne_parameter_space.py). Homodyne's ParameterSpace/
ParameterManager have no tied/DOF-reduction mechanism at all, and
ConfigManager._validate_config only warns on unknown TOP-LEVEL keys -- so a
tied_parameters entry under initial_parameters must be rejected explicitly by
ParameterManager, or it would be silently accepted and silently ignored
(the parameter stays independently free instead of constrained).
"""

from __future__ import annotations

import pytest

from xpcsjax.config.parameter_manager import ParameterManager
from xpcsjax.config.parameter_registry import AnalysisMode


@pytest.mark.parametrize(
    "analysis_mode",
    [
        AnalysisMode.STATIC_ANISOTROPIC,
        AnalysisMode.STATIC_ISOTROPIC,
        AnalysisMode.LAMINAR_FLOW,
    ],
)
def test_tied_parameters_rejected_for_homodyne_modes(analysis_mode):
    config = {
        "analysis_mode": str(analysis_mode.value),
        "initial_parameters": {"tied_parameters": {"D_offset": "D0"}},
    }
    with pytest.raises(ValueError, match="tied_parameters"):
        ParameterManager(config_dict=config, analysis_mode=analysis_mode)


def test_tied_parameters_absent_is_noop_for_homodyne():
    config = {"analysis_mode": "laminar_flow", "initial_parameters": {}}
    pm = ParameterManager(config_dict=config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    assert pm.get_active_parameters()  # constructs without error


def test_empty_tied_parameters_dict_is_noop_for_homodyne():
    config = {
        "analysis_mode": "laminar_flow",
        "initial_parameters": {"tied_parameters": {}},
    }
    ParameterManager(config_dict=config, analysis_mode=AnalysisMode.LAMINAR_FLOW)


def test_tied_parameters_still_allowed_for_two_component():
    config = {
        "analysis_mode": "two_component",
        "initial_parameters": {"tied_parameters": {"D_offset": "D0"}},
    }
    # Homodyne ParameterManager doesn't validate physics-parameter names (that
    # is heterodyne_parameter_space.py's job); it only guards the cross-mode
    # bypass, so two_component must not raise here.
    ParameterManager(config_dict=config, analysis_mode=AnalysisMode.TWO_COMPONENT)
