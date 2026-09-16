"""C9: the per-angle contrast/offset override parser used to be two
byte-identical ~35-line copies in wrapper.py (plain path + sequential path),
differing only in the angle-count variable name and log wording. This pins
the shared :func:`parse_per_angle_scaling_overrides` reproduces both call
sites' exact original behavior and log text.
"""

from __future__ import annotations

import numpy as np

from xpcsjax.optimization.nlsq.parameter_utils import (
    parse_per_angle_scaling_overrides,
)


class _RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)


def test_none_input_yields_no_overrides():
    contrast, offset = parse_per_angle_scaling_overrides(None, 5, _RecordingLogger())
    assert contrast is None and offset is None


def test_valid_overrides_pass_through():
    logger = _RecordingLogger()
    contrast, offset = parse_per_angle_scaling_overrides(
        {"contrast": [0.1, 0.2, 0.3], "offset": [1.0, 1.1, 1.2]}, 3, logger
    )
    np.testing.assert_allclose(contrast, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(offset, [1.0, 1.1, 1.2])
    assert logger.warnings == []


def test_wrong_length_is_dropped_with_default_wording():
    logger = _RecordingLogger()
    contrast, offset = parse_per_angle_scaling_overrides({"contrast": [0.1, 0.2]}, 3, logger)
    assert contrast is None and offset is None
    assert logger.warnings == [
        "per_angle_scaling contrast override has 2 entries (expected 3); ignoring override"
    ]


def test_malformed_value_is_dropped_with_default_wording():
    logger = _RecordingLogger()
    contrast, _ = parse_per_angle_scaling_overrides({"contrast": "not-a-number"}, 3, logger)
    assert contrast is None
    assert logger.warnings == ["Invalid per-angle contrast override; ignoring"]


def test_sequential_wording_matches_original_second_call_site():
    logger = _RecordingLogger()
    parse_per_angle_scaling_overrides(
        {"contrast": [0.1, 0.2]},
        3,
        logger,
        mismatch_scope="Sequential per-angle",
        invalid_prefix="sequential ",
    )
    assert logger.warnings == [
        "Sequential per-angle contrast override has 2 entries (expected 3); ignoring override"
    ]

    logger2 = _RecordingLogger()
    parse_per_angle_scaling_overrides(
        {"offset": object()},
        3,
        logger2,
        mismatch_scope="Sequential per-angle",
        invalid_prefix="sequential ",
    )
    assert logger2.warnings == ["Invalid sequential per-angle offset override; ignoring"]
