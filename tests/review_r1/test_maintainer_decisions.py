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


# ------------------------------------------------------------ data-B1 / gui-B2
def _per_row_q_data():
    return {
        "phi_angles_list": np.array([0.0, 90.0, 180.0]),
        "c2_exp": np.ones((3, 4, 4)),
        "wavevector_q_list": np.array([0.010, 0.011, 0.012]),
        "t1": np.arange(4.0),
        "t2": np.arange(4.0),
    }


def test_data_b1_per_row_q_follows_the_kept_angles():
    from xpcsjax.data.angle_filtering import apply_angle_filtering_for_optimization

    config = {
        "phi_filtering": {"enabled": True, "target_ranges": [{"min_angle": 80, "max_angle": 100}]}
    }
    out = apply_angle_filtering_for_optimization(_per_row_q_data(), config)
    np.testing.assert_allclose(out["phi_angles_list"], [90.0])
    # The fit reads wavevector_q_list[0]: it must be the q of a row that was kept.
    np.testing.assert_allclose(out["wavevector_q_list"], [0.011])


def test_data_b1_single_entry_q_is_left_alone():
    from xpcsjax.data.angle_filtering import apply_angle_filtering_for_optimization

    data = _per_row_q_data()
    data["wavevector_q_list"] = np.array([0.010])
    config = {
        "phi_filtering": {"enabled": True, "target_ranges": [{"min_angle": 80, "max_angle": 100}]}
    }
    out = apply_angle_filtering_for_optimization(data, config)
    np.testing.assert_allclose(out["wavevector_q_list"], [0.010])


def test_gui_b2_service_phi_subset_slices_per_row_q():
    from xpcsjax.service.data import _subset_data_by_phi

    data = _per_row_q_data()
    _subset_data_by_phi(data, np.asarray(data["phi_angles_list"]), [90.0])
    np.testing.assert_allclose(data["phi_angles_list"], [90.0])
    np.testing.assert_allclose(data["wavevector_q_list"], [0.011])


# ---------------------------------------------------------------------- core-B9
def test_core_b9_reported_fraction_is_the_one_the_kernel_uses():
    import jax.numpy as jnp

    from xpcsjax.core.heterodyne_models import TwoComponentModel
    from xpcsjax.core.heterodyne_physics_kernel import _fraction

    params = np.zeros(14)  # f0 = f1 = f2 = f3 = 0: on the lower bound
    t = jnp.arange(1.0, 5.0)
    reported = np.asarray(TwoComponentModel().compute_fraction(params, t))
    used = np.asarray(_fraction(t, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_allclose(reported, used)
    assert np.all(reported > 0)  # the smooth bound, not a hard 0.0


# ------------------------------------------------------------------------ R3-A1
def _duplicate_phi_fixture():
    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component

    model, c2, phi = make_synthetic_two_component(n_phi=4, n_t=12)
    phi = np.array(phi, copy=True)
    phi[-1] = phi[0]
    return model, c2, phi


def test_r3_a1_stratified_labels_match_shared_slots_for_duplicate_phi(monkeypatch):
    from xpcsjax.optimization.nlsq.heterodyne_adapter import NLSQAdapter
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_results import NLSQResult
    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
        fit_heterodyne_stratified_least_squares as fit_strat,
    )

    def solve(self, *, initial_params, **kwargs):
        return NLSQResult(np.asarray(initial_params), self._parameter_names, True, "ok")

    monkeypatch.setattr(NLSQAdapter, "fit", solve)
    model, c2, phi = _duplicate_phi_fixture()
    result = fit_strat(
        model=model, c2=c2, phi=phi, weights=None, config=NLSQConfig(per_angle_mode="individual")
    )
    diag = result.nlsq_diagnostics
    np.testing.assert_array_equal(diag["phi_angles"], np.unique(phi))
    assert len(diag["chi2_per_angle"]) == len(diag["phi_angles"])


def test_r3_a1_engine_route_declines_duplicate_phi():
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_engine_route import fit_two_component_via_engine

    model, c2, phi = _duplicate_phi_fixture()
    with pytest.raises(NotImplementedError, match="duplicate phi"):
        fit_two_component_via_engine(model, c2, phi, NLSQConfig(per_angle_mode="individual"))


# ---------------------------------------------------------------------- cli-B10
def test_cli_b10_package_no_longer_exports_an_ambiguous_main():
    import subprocess
    import sys

    # Fresh process: a lazy ``main`` export used to resolve to the submodule.
    code = (
        "import xpcsjax.cli as c\n"
        "assert 'main' not in c.__all__\n"
        "from xpcsjax.cli.main import main\n"
        "assert callable(main)\n"
        "from xpcsjax.cli import run_nlsq\n"
        "assert callable(run_nlsq)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-400:]


# ------------------------------------------------------------------------ R4-C2
def test_r4_c2_allocation_guard_budgets_the_largest_candidate(monkeypatch):
    import io

    import h5py

    from tests.review_r1.test_codex_data import _half, _write_aps_u
    from xpcsjax.data import xpcs_loader
    from xpcsjax.data.hdf5_readers import guard_aps_u_intermediate_allocation

    buf = io.BytesIO()
    _write_aps_u(buf, [_half(1.0), _half(1.0, n_t=8)], [0.0, 90.0])
    buf.seek(0)
    monkeypatch.setattr(xpcs_loader, "MAX_CORRELATION_ALLOC_BYTES", 256)
    with h5py.File(buf, "r") as f:
        groups = f["xpcs/twotime/correlation_map"]
        with pytest.raises(xpcs_loader.XPCSDataFormatError):
            guard_aps_u_intermediate_allocation(
                groups, sorted(groups), [0, 1], source="candidate probe"
            )


# ------------------------------------------------------------------- optwrap-B4
def test_optwrap_b4_averaged_covariance_uses_the_expanded_dof(monkeypatch):
    from tests.optimization.test_fixed_parameters_integration import _config, _synthetic_data
    from xpcsjax.config import ConfigManager
    from xpcsjax.optimization.nlsq.core import fit_nlsq_jax
    from xpcsjax.optimization.nlsq.wrapper import NLSQWrapper

    seen = {}
    original = NLSQWrapper._post_process_results

    def spy(self, *args, **kwargs):
        seen["pcov"] = np.array(kwargs["pcov"], copy=True)
        seen["n_data"] = kwargs["n_data"]
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NLSQWrapper, "_post_process_results", spy)
    data = _synthetic_data("laminar_flow")  # 3 angles -> auto resolves to averaged
    result = fit_nlsq_jax(
        data, ConfigManager(config_override=_config("laminar_flow")), use_adapter=False
    )
    assert result.nlsq_diagnostics["per_angle_mode"] == "averaged"
    n, p_solver, p_expanded = seen["n_data"], seen["pcov"].shape[0], len(result.parameters)
    assert (p_solver, p_expanded) == (9, 13)
    # Physics tail: the solver covariance, re-expressed on the expanded DOF.
    np.testing.assert_allclose(
        np.asarray(result.covariance)[-1, -1],
        seen["pcov"][-1, -1] * (n - p_solver) / (n - p_expanded),
        rtol=1e-12,
    )


# ------------------------------------------------------------------------ R2-A4
def test_r2_a4_unconverged_search_plus_unconverged_refinement_is_not_converged():
    from unittest.mock import patch

    from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESWrapper, CMAESWrapperConfig

    class Search:
        def __init__(self, config):
            pass

        def fit(self, **kw):
            return {
                "popt": kw["p0"],
                "cmaes_diagnostics": {"convergence_reason": "max_generations"},
            }

    x = np.arange(1.0, 5.0)
    cfg = CMAESWrapperConfig(
        refinement_workflow="standard", refinement_max_nfev=1, auto_memory=False
    )
    with patch("nlsq.global_optimization.CMAESOptimizer", Search):
        r = CMAESWrapper(cfg).fit(
            lambda x, a: a * x, x, 2 * x, np.array([1.0]), (np.array([0.0]), np.array([4.0]))
        )
    assert r.nlsq_refined and r.diagnostics["refinement_converged"] is False
    assert r.success  # still an eligible candidate for the keep-better callers
    assert r.diagnostics["converged"] is False  # but not reported as converged


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
