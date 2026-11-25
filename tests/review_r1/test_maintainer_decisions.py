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


# --------------------------------------- optescape-B3 / optstrat-B10 (dead streaming)
def test_streaming_refinement_is_skipped_explicitly(caplog):
    from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESWrapper, CMAESWrapperConfig

    x = np.arange(1.0, 5.0)
    p0 = np.array([1.0])
    with caplog.at_level("WARNING"):
        result = CMAESWrapper(
            CMAESWrapperConfig(refinement_workflow="streaming")
        )._run_nlsq_refinement(lambda x, a: a * x, x, 2 * x, p0, (np.array([0.0]), np.array([4.0])))
    # Same outcome as before (CMA-ES point kept, nothing adopted), now stated plainly.
    assert result["adoptable"] is False and result["success"] is False
    assert result["popt"] is p0 and result["chi_squared"] is None
    assert "not supported" in result["mesg"] and "not supported" in caplog.text


def test_fallback_chain_names_the_unsupported_streaming_tier():
    import inspect

    from xpcsjax.optimization.nlsq import fallback_chain

    src = inspect.getsource(fallback_chain)
    # The tier used to die with an AttributeError on ConfigManager.hybrid_normalize.
    assert "generic hybrid-streaming tier is not supported" in src


# ------------------------------------------------- opthet-B10 / optstrat-B9 (L3 enable)
def _ad_block(enable):
    reg = {"mode": "relative", "lambda": 1.0}
    if enable is not None:
        reg["enable"] = enable
    return {"enable": True, "per_angle_mode": "auto", "regularization": reg}


@pytest.mark.parametrize("enable, l3_on", [(None, True), (True, True), (False, False)])
def test_l3_enable_flag_controls_the_shared_controller(enable, l3_on):
    from xpcsjax.optimization.nlsq.anti_degeneracy_controller import AntiDegeneracyController

    ctrl = AntiDegeneracyController.from_config(
        config_dict=_ad_block(enable),
        n_phi=3,
        phi_angles=np.deg2rad(np.array([0.0, 60.0, 120.0])),
        n_physical=7,
    )
    assert (ctrl.regularizer is not None) is l3_on


@pytest.mark.parametrize("enable, mode", [(None, "relative"), (True, "relative"), (False, "none")])
def test_l3_enable_flag_controls_the_heterodyne_config(enable, mode):
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig

    cfg = NLSQConfig.from_dict({"anti_degeneracy": _ad_block(enable)})
    assert cfg.regularization_mode == mode


@pytest.mark.parametrize(
    "template", ["static_anisotropic", "static_isotropic", "laminar_flow", "two_component"]
)
def test_shipped_templates_keep_layer_3_on(template):
    """Templates used to ship ``enable: false`` while the flag was ignored (L3 ran)."""
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[2] / "xpcsjax" / "config" / "templates"
    cfg = yaml.safe_load((path / f"xpcsjax_{template}.yaml").read_text(encoding="utf-8"))
    reg = cfg["optimization"]["nlsq"]["anti_degeneracy"]["regularization"]
    assert reg["enable"] is True and reg["mode"] != "none"
