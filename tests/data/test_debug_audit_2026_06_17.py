"""Regression tests for the 2026-06-17 debug-audit fixes (data/utils layer).

Each test fails against the pre-fix code and passes after the fix.
"""

from __future__ import annotations

import json
import logging

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


# ---------------------------------------------------------------------------
# AsyncWriter._write_json — non-finite floats must serialize to strict JSON
# (finding #23).
# ---------------------------------------------------------------------------
def test_write_json_emits_strict_json_for_non_finite(tmp_path) -> None:
    from xpcsjax.utils.async_io import AsyncWriter

    path = tmp_path / "out.json"
    AsyncWriter._write_json(path, {"a": float("nan"), "b": float("inf"), "c": 1.0})

    text = path.read_text()
    # Bare NaN/Infinity tokens are invalid JSON; strict parsing must succeed.
    parsed = json.loads(text)  # strict by default
    assert parsed["a"] is None
    assert parsed["c"] == 1.0


# ---------------------------------------------------------------------------
# PrefetchLoader must yield a legitimate None item, not assert on it
# (finding #25).
# ---------------------------------------------------------------------------
def test_prefetch_loader_yields_none_items() -> None:
    from xpcsjax.utils.async_io import PrefetchLoader

    loader = PrefetchLoader(iter([1, 2, 3]), lambda _x: None)
    items = list(loader)
    assert items == [None, None, None]


# ---------------------------------------------------------------------------
# AnalysisSummaryLogger.log_summary must not crash on a non-float metric
# (finding #24).
# ---------------------------------------------------------------------------
def test_log_summary_survives_non_float_metric(caplog) -> None:
    from xpcsjax.config.parameter_registry import AnalysisMode
    from xpcsjax.utils.logging import AnalysisSummaryLogger

    summary = AnalysisSummaryLogger("run-xyz", AnalysisMode.LAMINAR_FLOW)
    summary.record_metric("good", 1.23456789)
    summary.record_metric("bad", "not-a-number")  # type: ignore[arg-type]

    logger = logging.getLogger("test.summary")
    # Must not raise despite the non-numeric metric.
    summary.log_summary(logger)
