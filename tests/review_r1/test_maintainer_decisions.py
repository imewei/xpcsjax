"""Tests for the maintainer-decision items fixed after PR #80.

Item IDs refer to ``docs/reviews/2026-11-24-claude-codex-review-loop.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel

pytestmark = [pytest.mark.filterwarnings("ignore:Ill-conditioned Jacobian:UserWarning")]


# ---------------------------------------------------------------------- core-B1
def _model(n_frames: int = 100, dt: float = 0.1) -> HeterodyneModel:
    ap = {"dt": dt, "start_frame": 1, "end_frame": n_frames, "scattering": {"wavevector_q": 0.01}}
    return HeterodyneModel.from_config({"analyzer_parameters": ap})


def test_core_b1_tail_clamp_keeps_the_leading_model_times():
    # Config asks for 100 frames, the file has 80: the loader clamps the END, so
    # the model must keep t = dt..80dt (not drop its first 20 frames).
    model = _model(100)
    model.sync_time_axis(np.arange(80.0), n_tail_clamped=20)
    np.testing.assert_allclose(np.asarray(model.t), np.arange(1, 81) * 0.1)


def test_core_b1_leading_exclusion_is_unchanged():
    # No clamp recorded: a shorter data axis still means leading frames were dropped.
    model = _model(100)
    model.sync_time_axis(np.arange(99.0))
    np.testing.assert_allclose(np.asarray(model.t), np.arange(2, 101) * 0.1)


def test_core_b1_clamp_and_leading_exclusion_combine():
    model = _model(100)
    model.sync_time_axis(np.arange(79.0), n_tail_clamped=20)
    np.testing.assert_allclose(np.asarray(model.t), np.arange(2, 81) * 0.1)


def test_core_b1_loader_reports_frames_clamped_at_end():
    from xpcsjax.data.xpcs_loader import XPCSDataLoader

    loader = XPCSDataLoader.__new__(XPCSDataLoader)
    loader.analyzer_config = {"dt": 0.1, "start_frame": 1, "end_frame": 100}
    c2 = loader._apply_frame_slicing_to_selected_q(np.ones((1, 80, 80)))
    assert c2.shape == (1, 80, 80)
    assert loader._frames_clamped_at_end() == 20
    loader.analyzer_config = {"dt": 0.1, "start_frame": 1, "end_frame": 80}
    loader._apply_frame_slicing_to_selected_q(np.ones((1, 80, 80)))
    assert loader._frames_clamped_at_end() == 0
