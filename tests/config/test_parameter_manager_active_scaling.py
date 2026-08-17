"""Regression test: get_active_parameters() must stay physics-only.

Bug: an explicit `initial_parameters.active_parameters` list that names
contrast/offset (bare or per-angle contrast_N/offset_N) leaked those scaling
names past ParameterManager.get_active_parameters(), breaking the
physics-only contract documented on the method (and relied on by
service/persist.py and cli/plot_families/simulated.py). Downstream code
building a params array from these names against a physics-only initial
value dict then hit a KeyError on 'contrast'.
"""

from xpcsjax.config.parameter_manager import ParameterManager
from xpcsjax.config.parameter_registry import AnalysisMode


def test_get_active_parameters_excludes_bare_scaling_names() -> None:
    config = {
        "initial_parameters": {
            "active_parameters": ["D0", "alpha", "contrast", "offset"],
        }
    }
    pm = ParameterManager(config, AnalysisMode.LAMINAR_FLOW)
    active = pm.get_active_parameters()
    assert active == ["D0", "alpha"]
    assert "contrast" not in active
    assert "offset" not in active


def test_get_active_parameters_excludes_per_angle_scaling_names() -> None:
    config = {
        "initial_parameters": {
            "active_parameters": ["D0", "alpha", "contrast_0", "offset_1"],
        }
    }
    pm = ParameterManager(config, AnalysisMode.LAMINAR_FLOW)
    active = pm.get_active_parameters()
    assert active == ["D0", "alpha"]
