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


# ------------------------------------------------------------------- opthet-B11
@pytest.mark.parametrize(
    "weight, loss, engine_runs",
    [
        (1.0, "soft_l1", True),  # unit weights: dropping them is exact
        (4.0, "linear", True),  # uniform + linear loss: a pure rescale, same argmin
        (4.0, "soft_l1", False),  # uniform + robust loss: the residual scale matters
    ],
)
def test_opthet_b11_engine_only_drops_weights_when_that_is_exact(
    monkeypatch, weight, loss, engine_runs
):
    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
    from xpcsjax.optimization.nlsq.heterodyne_adapter import NLSQAdapter
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_engine_route import fit_two_component_via_engine
    from xpcsjax.optimization.nlsq.heterodyne_results import NLSQResult

    def solve(self, *, initial_params, **kwargs):
        return NLSQResult(np.asarray(initial_params), self._parameter_names, True, "ok")

    monkeypatch.setattr(NLSQAdapter, "fit", solve)
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=12)
    cfg = NLSQConfig(loss=loss)
    weights = np.full_like(c2, weight)
    if engine_runs:
        assert fit_two_component_via_engine(model, c2, phi, cfg, weights=weights) is not None
    else:
        with pytest.raises(NotImplementedError, match="weights"):
            fit_two_component_via_engine(model, c2, phi, cfg, weights=weights)


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
