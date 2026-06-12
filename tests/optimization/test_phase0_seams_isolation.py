# tests/optimization/test_phase0_seams_isolation.py
"""Phase-0 isolation gate: the new seams are pure, additive, and not yet wired in.

Guards that Phase 0 introduced ZERO call-site changes — no production module under
xpcsjax/optimization/nlsq imports per_angle_mode yet, EXCEPT parameter_index_mapper
(which legitimately gained the canonical classmethod).
"""

from __future__ import annotations

import pathlib

import xpcsjax.optimization.nlsq.per_angle_mode as pam
from xpcsjax.optimization.nlsq.parameter_index_mapper import ParameterIndexMapper


def test_public_surface_exists():
    assert hasattr(pam, "PerAngleMode")
    assert hasattr(pam, "DEFAULT_CONSTANT_SCALING_THRESHOLD")
    assert callable(pam.resolve_per_angle_mode)
    assert callable(pam.n_optimized)
    assert callable(pam.PerAngleScalingPlan)
    assert callable(ParameterIndexMapper.canonical)


def test_no_callsite_imports_per_angle_mode_yet():
    # Phase 0 is pure-seam: only parameter_index_mapper may import per_angle_mode.
    nlsq_dir = pathlib.Path(pam.__file__).parent
    offenders = []
    allowed = {"per_angle_mode.py", "parameter_index_mapper.py"}
    for path in nlsq_dir.glob("*.py"):
        if path.name in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if "per_angle_mode import" in text or "import per_angle_mode" in text:
            offenders.append(path.name)
    assert offenders == [], f"Phase-0 call-site drift: {offenders} import per_angle_mode"
