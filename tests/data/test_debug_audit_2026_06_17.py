"""Regression tests for the 2026-06-17 debug-audit fixes (data/utils layer).

Each test fails against the pre-fix code and passes after the fix.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# validators.validate_frame_range — lower bound must be checked even when
# end_frame is None (finding #15).
# ---------------------------------------------------------------------------
def test_frame_range_lower_bound_checked_when_end_is_none() -> None:
    from xpcsjax.data.validators import validate_frame_range

    errors = validate_frame_range(0, None, min_frame=1)
    assert any("must be >=" in e for e in errors), errors


def test_frame_range_valid_start_with_end_none_passes() -> None:
    from xpcsjax.data.validators import validate_frame_range

    assert validate_frame_range(5, None, min_frame=1) == []


# ---------------------------------------------------------------------------
# angle_filtering.normalize_angle_to_symmetric_range — -180 must stay -180
# (finding #16).
# ---------------------------------------------------------------------------
def test_normalize_angle_preserves_minus_180_array() -> None:
    from xpcsjax.data.angle_filtering import normalize_angle_to_symmetric_range

    out = normalize_angle_to_symmetric_range(np.array([180.0, -180.0, 360.0]))
    np.testing.assert_array_equal(out, np.array([180.0, -180.0, 0.0]))


def test_normalize_angle_preserves_minus_180_scalar() -> None:
    from xpcsjax.data.angle_filtering import normalize_angle_to_symmetric_range

    assert normalize_angle_to_symmetric_range(-180.0) == -180.0
    # Out-of-range folding still works.
    assert normalize_angle_to_symmetric_range(210.0) == -150.0
    assert normalize_angle_to_symmetric_range(-210.0) == 150.0
