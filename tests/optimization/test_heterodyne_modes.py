"""Tests for heterodyne per-angle mode vocabulary parity with homodyne."""
from __future__ import annotations

import pytest

from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig


def test_individual_mode_accepted() -> None:
    """`individual` is the canonical name (matches homodyne docs)."""
    cfg = NLSQConfig(per_angle_mode="individual")
    assert cfg.per_angle_mode == "individual"
    errors = cfg.validate()
    assert errors == [], f"expected no validation errors, got {errors}"


def test_independent_deprecation_alias() -> None:
    """`independent` maps to `individual` with a DeprecationWarning that points at the user's call site."""
    with pytest.warns(DeprecationWarning, match=r"'independent' is deprecated") as records:
        cfg = NLSQConfig(per_angle_mode="independent")  # type: ignore[arg-type]
    assert cfg.per_angle_mode == "individual"
    assert len(records) == 1
    # stacklevel should point at this test file, not dataclass-synthesized <string> code
    assert records[0].filename.endswith("test_heterodyne_modes.py"), (
        f"DeprecationWarning fired at {records[0].filename}:{records[0].lineno} — "
        "expected to point at user call site (stacklevel issue?)"
    )


def test_averaged_function_renamed() -> None:
    """The averaged-scaling joint solver uses the corrected name."""
    from xpcsjax.optimization.nlsq import heterodyne_core

    assert hasattr(heterodyne_core, "_fit_joint_averaged_multi_phi"), (
        "expected renamed function"
    )
    assert not hasattr(heterodyne_core, "_fit_joint_constant_multi_phi"), (
        "old mislabeled name must be removed — "
        "true 'constant' mode lands in Sub-PR B with its own dedicated function"
    )
