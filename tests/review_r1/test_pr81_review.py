"""Regression tests for the Codex + Antigravity review of PR #81.

Each test reproduces one confirmed finding from that review; the IDs in the
test names are the reviewers' (``c`` = Codex, ``a`` = Antigravity).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig


# ------------------------------------------------------------------ c-F1 / a-F4
@pytest.mark.parametrize("fill", ["zeros", "mask_with_tiny"])
def test_engine_route_declines_non_positive_weights_under_linear_loss(fill):
    from xpcsjax.optimization.nlsq.heterodyne_engine_route import fit_two_component_via_engine

    model, c2, phi = make_synthetic_two_component(n_phi=4, n_t=12)
    weights = np.zeros_like(c2)
    if fill == "mask_with_tiny":
        weights[:, 6:, :] = 1e-9  # a mask np.allclose's absolute tolerance calls uniform
    with pytest.raises(NotImplementedError):
        fit_two_component_via_engine(model, c2, phi, NLSQConfig(loss="linear"), weights=weights)


# ------------------------------------------------------------------------- a-F1
@pytest.mark.parametrize("mode", ["individual", "auto", "constant"])
def test_masked_fit_with_no_dof_returns_an_undefined_statistic(mode):
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=10)
    weights = np.zeros_like(c2)
    weights[0, 1, 2:7] = 1.0  # 5 observations for more than 5 parameters
    config = NLSQConfig(per_angle_mode=mode, ftol=1e-1, xtol=1e-1, gtol=1e-1)
    result = fit_nlsq_multi_phi(model, c2, phi, config, weights=weights)
    assert np.isnan(result.reduced_chi_squared)
    assert result.quality_flag != "good"
    assert np.isnan(result.uncertainties).all()
    assert result.nlsq_diagnostics["covariance_is_placeholder"] is True


# ------------------------------------------------------------------------- c-F6
def test_clamp_metadata_accepts_numpy_integer_frame_counts():
    from xpcsjax.data.xpcs_loader import XPCSDataLoader

    loader = XPCSDataLoader.__new__(XPCSDataLoader)
    loader.analyzer_config = {"start_frame": 1, "end_frame": np.int64(100), "dt": 0.1}
    loader._applied_end_frame = np.int64(80)
    clamped = loader._frames_clamped_at_end()
    assert clamped == 20 and type(clamped) is int
    loader.analyzer_config["end_frame"] = True  # a bool is not a frame count
    assert loader._frames_clamped_at_end() == 0


# ------------------------------------------------------------------ c-F2 / a-F2
@pytest.mark.parametrize("adapter_covariance", [True, False])
def test_stratified_masked_fit_with_no_dof_is_undefined(monkeypatch, adapter_covariance):
    from xpcsjax.optimization.nlsq.heterodyne_adapter import NLSQAdapter
    from xpcsjax.optimization.nlsq.heterodyne_results import NLSQResult
    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
        fit_heterodyne_stratified_least_squares as fit_strat,
    )

    def solve(self, *, initial_params, **kwargs):
        x = np.asarray(initial_params)
        cov = np.eye(x.size) if adapter_covariance else None
        return NLSQResult(x, self._parameter_names, True, "ok", covariance=cov)

    monkeypatch.setattr(NLSQAdapter, "fit", solve)
    model, c2, phi = make_synthetic_two_component(n_phi=4, n_t=12)
    weights = np.zeros_like(c2)
    weights[0, 1, 2:7] = 1.0  # 5 observations
    result = fit_strat(
        model=model,
        c2=c2,
        phi=phi,
        weights=weights,
        config=NLSQConfig(per_angle_mode="individual"),
    )
    assert np.isnan(result.reduced_chi_squared)
    assert result.quality_flag != "good"
    assert np.isnan(result.covariance).all()
    assert result.nlsq_diagnostics["covariance_is_placeholder"] is True


def test_result_builder_reports_undefined_reduced_chi2_only_when_masked():
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
    from xpcsjax.optimization.nlsq.heterodyne_result_builder import build_hybrid_streaming_result

    model = HeterodyneModel.from_config(
        {
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 12,
                "scattering": {"wavevector_q": 0.01},
            }
        }
    )
    p = np.asarray(model.param_manager.get_initial_values())
    phi = np.array([0.0, 90.0, 135.0])

    def build(**info):
        return build_hybrid_streaming_result(
            model=model, popt=p, pcov=np.eye(len(p)), info={"ssr": 1.0, **info}, phi_angles=phi
        )

    assert np.isnan(build(n_data_points=len(p), n_data_rows=10 * len(p)).reduced_chi_squared)
    assert np.isfinite(build(n_data_points=len(p)).reduced_chi_squared)  # unmasked: legacy floor


# ------------------------------------------------------------------------- c-F3
def test_duplicate_phi_rows_keep_their_own_mask_on_the_pointwise_paths():
    from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
        build_heterodyne_stratified_data,
    )
    from xpcsjax.optimization.nlsq.strategies.heterodyne_hybrid_streaming import (
        build_heterodyne_pointwise_model,
    )

    model, c2, _ = make_synthetic_two_component(n_phi=4, n_t=12)
    weights = np.ones_like(c2)
    weights[0] = 0.0  # mask only the FIRST of the two measurements at phi = 0
    data = build_heterodyne_stratified_data(model, c2, np.array([0.0, 0.0, 90.0, 135.0]), weights)
    meta = build_heterodyne_pointwise_model(
        stratified_data=data,
        model=model,
        physical_param_names=list(model.param_manager.varying_names),
    )[-1]
    assert meta["n_contributing_points"] == 3 * 11 * 10
    # The second phi = 0 measurement and both other angles keep finite sigma.
    assert int(np.isfinite(meta["sigma"]).sum()) == 3 * 11 * 10


# ------------------------------------------------------------------ c-F4 / a-F3
def test_streaming_ssr_ignores_masked_pixels():
    from xpcsjax.optimization.nlsq.strategies.heterodyne_hybrid_streaming import _observed_ssr

    y, pred = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.5, 13.0])
    assert _observed_ssr(y, pred, None) == pytest.approx(0.25 + 100.0)
    assert _observed_ssr(y, pred, np.array([1.0, 2.0, np.inf])) == pytest.approx(0.25)


# ------------------------------------------------------------------ c-F5 / a-F5
def test_nested_l3_off_switch_beats_a_flat_regularization_mode():
    off = {"regularization": {"enable": False, "mode": "relative"}}
    cfg = NLSQConfig.from_dict({"regularization_mode": "adaptive", "anti_degeneracy": off})
    assert cfg.regularization_mode == "none"
    # Without the nested switch the flat key still wins, as for every other field.
    on = {"regularization": {"mode": "relative"}}
    cfg = NLSQConfig.from_dict({"regularization_mode": "adaptive", "anti_degeneracy": on})
    assert cfg.regularization_mode == "adaptive"


def test_controller_callbacks_carry_no_l3_groups_when_l3_is_off():
    from xpcsjax.optimization.nlsq.anti_degeneracy_controller import AntiDegeneracyController

    def callbacks(regularization):
        ctrl = AntiDegeneracyController.from_config(
            {"per_angle_mode": "individual", "regularization": regularization},
            n_phi=4,
            phi_angles=np.array([0.0, 45.0, 90.0, 135.0]),
            n_physical=7,
        )
        return ctrl.create_nlsq_callbacks()

    assert "group_variance_indices" in callbacks({"enable": True, "mode": "relative"})
    assert "group_variance_indices" not in callbacks({"enable": False, "mode": "relative"})


def test_l3_gate_reads_enable_and_mode():
    # The builders that use this gate are exercised in test_pr81_six_pass.py
    # (laminar) and below (heterodyne streaming).
    from xpcsjax.optimization.nlsq.strategies.hybrid_streaming_layers import l3_is_enabled

    assert l3_is_enabled({}) and l3_is_enabled({"mode": "relative"})
    assert not l3_is_enabled({"enable": False, "mode": "relative"})
    assert not l3_is_enabled({"mode": "none"})


# ------------------------------------------------- Codex closing check, new F1
@pytest.mark.parametrize(("enable", "expected"), [(False, False), (None, False), (True, True)])
def test_heterodyne_streaming_keeps_treating_a_null_l3_enable_as_off(enable, expected):
    from unittest.mock import patch

    from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
        build_heterodyne_stratified_data,
    )
    from xpcsjax.optimization.nlsq.strategies import heterodyne_hybrid_streaming as stream

    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=8)
    pm = model.param_manager
    seen = {}

    class Capture:
        def __init__(self, cfg):
            seen["l3"] = bool(cfg.enable_group_variance_regularization)

        def fit(self, **kwargs):
            p = kwargs["p0"]
            return {"x": p, "pcov": np.eye(len(p)), "success": True}

    ad = {
        "per_angle_mode": "individual",
        "hierarchical": {"enable": False},
        "gradient_monitoring": {"enable": False},
        "regularization": {"enable": enable, "mode": "relative"},
    }
    with patch.object(stream, "AdaptiveHybridStreamingOptimizer", Capture):
        stream.fit_with_stratified_hybrid_streaming_heterodyne(
            stratified_data=build_heterodyne_stratified_data(model, c2, phi, None),
            model=model,
            physical_param_names=list(pm.varying_names),
            initial_params=pm.get_initial_values(),
            bounds=pm.get_bounds(),
            anti_degeneracy_config=ad,
        )
    assert seen["l3"] is expected


def test_streaming_masked_fit_with_no_dof_has_no_covariance():
    from unittest.mock import patch

    from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
        build_heterodyne_stratified_data,
    )
    from xpcsjax.optimization.nlsq.strategies import heterodyne_hybrid_streaming as stream

    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=8)
    pm = model.param_manager

    class Solver:
        def __init__(self, cfg):
            pass

        def fit(self, **kwargs):
            p = kwargs["p0"]
            return {"x": p, "pcov": np.eye(len(p)), "success": True}

    def fit(weights):
        with patch.object(stream, "AdaptiveHybridStreamingOptimizer", Solver):
            return stream.fit_with_stratified_hybrid_streaming_heterodyne(
                stratified_data=build_heterodyne_stratified_data(model, c2, phi, weights),
                model=model,
                physical_param_names=list(pm.varying_names),
                initial_params=pm.get_initial_values(),
                bounds=pm.get_bounds(),
                anti_degeneracy_config={"per_angle_mode": "constant"},
            )

    _, cov, info = fit(None)
    assert np.isfinite(cov).all() and info["covariance_is_placeholder"] is False

    weights = np.zeros_like(c2)
    weights[0, 1, 2:6] = 1.0  # 4 observations
    _, cov, info = fit(weights)
    assert np.isnan(cov).all() and info["covariance_is_placeholder"] is True
