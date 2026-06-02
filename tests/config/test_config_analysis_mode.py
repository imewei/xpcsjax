"""ConfigManager exposes a typed analysis_mode (quality-gate F4).

The mode lived only as a string under the untrusted-YAML ``config`` dict, so
callers scattered ``config.config.get("analysis_mode", "")`` lookups. A typed
``analysis_mode`` property returns the validated :class:`AnalysisMode` enum.
"""

from __future__ import annotations

from xpcsjax.config import ConfigManager
from xpcsjax.config.parameter_registry import AnalysisMode


def test_analysis_mode_property_returns_enum():
    cm = ConfigManager(config_override={"analysis_mode": "laminar_flow"})
    assert cm.analysis_mode == AnalysisMode.LAMINAR_FLOW
    assert isinstance(cm.analysis_mode, AnalysisMode)
    # StrEnum keeps string comparisons working for existing callers.
    assert cm.analysis_mode == "laminar_flow"
