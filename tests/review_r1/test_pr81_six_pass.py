"""Tests added after the six-pass review of PR #81.

Most close a coverage gap that review found: a fix that was only checked
through a helper or a diagnostic count is checked here through the value the
fit actually reports.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig

N_T = 12
MASKED_ROWS = slice(1, 6)  # 5 rows of 10 off-diagonal pixels, per angle


def _masked(c2):
    weights = np.ones_like(c2)
    weights[:, MASKED_ROWS, :] = 0.0
    return weights


def _stub_adapter(monkeypatch, covariance=None):
    """Make the heterodyne adapter return its start point (and an optional covariance)."""
    from xpcsjax.optimization.nlsq.heterodyne_adapter import NLSQAdapter
    from xpcsjax.optimization.nlsq.heterodyne_results import NLSQResult

    def solve(self, *, initial_params, **kwargs):
        x = np.asarray(initial_params)
        cov = None if covariance is None else covariance(x.size)
        return NLSQResult(x, self._parameter_names, True, "ok", covariance=cov)

    monkeypatch.setattr(NLSQAdapter, "fit", solve)


# ------------------------------------------------- R4-A3: the reported covariance
@pytest.mark.parametrize("mode", ["individual", "auto", "constant"])
def test_masked_joint_fit_rescales_the_reported_covariance(monkeypatch, mode):
    from xpcsjax.optimization.nlsq import heterodyne_data_prep
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    real = heterodyne_data_prep.covariance_on_valid_rows
    calls = []

    def spy(covariance, uncertainties, **counts):
        out = real(covariance, uncertainties, **counts)
        calls.append((counts, np.array(covariance, copy=True), out[0]))
        return out

    monkeypatch.setattr(heterodyne_data_prep, "covariance_on_valid_rows", spy)
    model, c2, phi = make_synthetic_two_component(n_phi=4, n_t=N_T)
    n_rows = 4 * (N_T - 1) * (N_T - 2)
    n_valid = n_rows - 4 * 5 * (N_T - 2)
    result = fit_nlsq_multi_phi(
        model, c2, phi, NLSQConfig(per_angle_mode=mode), weights=_masked(c2)
    )

    counts, solver_cov, rescaled = calls[-1]
    assert counts["n_valid"] == n_valid and counts["n_rows_solver"] >= n_rows
    scale = (counts["n_rows_solver"] - counts["n_params"]) / (n_valid - counts["n_params"])
    assert scale > 1.0
    np.testing.assert_allclose(rescaled, solver_cov * scale, equal_nan=True)
    # ... and it is the rescaled matrix that reaches the result.
    finite = np.isfinite(rescaled)
    assert finite.any()
    assert np.isin(rescaled[finite], np.asarray(result.covariance)).all()


def test_masked_stratified_fit_rescales_the_adapter_covariance(monkeypatch):
    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
        fit_heterodyne_stratified_least_squares as fit_strat,
    )

    _stub_adapter(monkeypatch, covariance=np.eye)
    model, c2, phi = make_synthetic_two_component(n_phi=4, n_t=N_T)
    result = fit_strat(
        model=model,
        c2=c2,
        phi=phi,
        weights=_masked(c2),
        config=NLSQConfig(per_angle_mode="individual"),
    )
    n_rows = 4 * (N_T - 1) * (N_T - 2)
    n_valid = n_rows - 4 * 5 * (N_T - 2)
    p = 2 * 4 + len(model.param_manager.varying_names)
    cov = np.asarray(result.covariance)
    np.testing.assert_allclose(np.diag(cov), (n_rows - p) / (n_valid - p))


def test_weighted_duplicate_phi_runs_through_stratified_ls(monkeypatch):
    from xpcsjax.optimization.nlsq.heterodyne_data_prep import far_lag_noise_variance
    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
        fit_heterodyne_stratified_least_squares as fit_strat,
    )

    _stub_adapter(monkeypatch)
    model, c2, _ = make_synthetic_two_component(n_phi=4, n_t=N_T)
    weights = np.ones_like(c2)
    weights[0] = 0.0  # mask only the first of the two phi = 0 measurements
    result = fit_strat(
        model=model,
        c2=c2,
        phi=np.array([0.0, 0.0, 90.0, 135.0]),
        weights=weights,
        config=NLSQConfig(per_angle_mode="individual"),
    )
    # 3 unique angles -> 2 * 3 scaling slots; 3 unmasked measurements are observed.
    p = 2 * 3 + len(model.param_manager.varying_names)
    implied_dof = result.chi_squared / (far_lag_noise_variance(c2) * result.reduced_chi_squared)
    assert round(implied_dof) == 3 * (N_T - 1) * (N_T - 2) - p


# --------------------------------------------- zero data dof: the exact quality flag
@pytest.mark.parametrize("mode", ["individual", "auto", "constant"])
def test_fully_masked_joint_fit_is_graded_unknown(mode):
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=10)
    weights = np.zeros_like(c2)
    weights[0, 1, 2:7] = 1.0
    config = NLSQConfig(per_angle_mode=mode, ftol=1e-1, xtol=1e-1, gtol=1e-1)
    assert fit_nlsq_multi_phi(model, c2, phi, config, weights=weights).quality_flag == "unknown"


# ------------------------------------------------ core-B1: cold load and cache hit
def test_tail_clamp_is_reported_on_the_cold_load_and_on_the_cache_hit(tmp_path):
    from tests.review_r1.test_codex_data import _half, _hdf_config, _write_aps_u
    from xpcsjax.data import load_xpcs_data

    _write_aps_u(tmp_path / "clamp.h5", [_half(1.2, n_t=8)], [0.0])
    cfg = _hdf_config(tmp_path, "clamp.h5")
    cfg["analyzer_parameters"]["end_frame"] = 20  # the file has 8 frames

    cold = load_xpcs_data(config_dict=cfg)
    assert list(tmp_path.glob("*.npz")), "the cold load must have written a cache"
    warm = load_xpcs_data(config_dict=cfg)
    for data in (cold, warm):
        assert np.asarray(data["c2_exp"]).shape[-1] == 8
        assert data["frames_clamped_at_end"] == 12


# ---------------------------------------- Layer 3 off switch: laminar layer builder
@pytest.mark.parametrize(
    ("regularization", "l3_on"),
    [
        ({"mode": "relative"}, True),
        ({"enable": False, "mode": "relative"}, False),
        ({"mode": "none"}, False),
    ],
)
def test_laminar_streaming_layer_builder_honours_the_l3_off_switch(regularization, l3_on):
    from xpcsjax.optimization.nlsq.adaptive_regularization import AdaptiveRegularizationConfig
    from xpcsjax.optimization.nlsq.hierarchical import HierarchicalOptimizer
    from xpcsjax.optimization.nlsq.shear_weighting import ShearSensitivityWeighting
    from xpcsjax.optimization.nlsq.strategies.hybrid_streaming_layers import (
        configure_streaming_layers,
    )

    n_phi, names = 4, ["D0", "alpha", "D_offset"]
    x0 = np.concatenate([np.full(n_phi, 0.3), np.full(n_phi, 1.0), [100.0, 0.5, 1.0]])
    setup = configure_streaming_layers(
        per_angle_mode_actual="individual",
        constant_scaling_threshold=3,
        per_angle_scaling=True,
        n_phi=n_phi,
        n_physical=3,
        n_physical_free=3,
        free_physical_names=names,
        physical_param_names=names,
        is_laminar_flow=False,
        phi_unique=np.array([0.0, 45.0, 90.0, 135.0]),
        initial_params=x0,
        bounds=(np.zeros_like(x0), np.full_like(x0, 1e4)),
        stratified_data=None,
        resolved_physical=None,
        ad_config={},
        hierarchical_config={"enable": False},
        regularization_config=regularization,
        gradient_monitoring_config={"enable": False},
        enable_group_variance_regularization=False,
        group_variance_lambda=1.0,
        logger=logging.getLogger("test_l3_builder"),
        HierarchicalOptimizer=HierarchicalOptimizer,
        AdaptiveRegularizationConfig=AdaptiveRegularizationConfig,
        ShearSensitivityWeighting=ShearSensitivityWeighting,
        compute_quantile_per_angle_scaling=None,
    )
    assert (setup.adaptive_regularizer is not None) is l3_on


def test_nested_l3_off_switch_warns_when_it_overrides_a_flat_mode(caplog):
    off = {"regularization": {"enable": False, "mode": "relative"}}
    with caplog.at_level(logging.WARNING):
        NLSQConfig.from_dict({"regularization_mode": "adaptive", "anti_degeneracy": off})
    assert "ignoring regularization_mode='adaptive'" in caplog.text


# --------------------------- engine route: a deliberate decline vs a real failure
def test_engine_route_declines_with_its_own_exception_type():
    from xpcsjax.optimization.exceptions import EngineRouteDeclined
    from xpcsjax.optimization.nlsq.heterodyne_engine_route import fit_two_component_via_engine

    assert issubclass(EngineRouteDeclined, NotImplementedError)
    model, c2, phi = make_synthetic_two_component(n_phi=4, n_t=N_T)
    with pytest.raises(EngineRouteDeclined):  # masking weights
        fit_two_component_via_engine(model, c2, phi, NLSQConfig(), weights=_masked(c2))
    duplicate = np.array([0.0, 0.0, 90.0, 135.0])
    with pytest.raises(EngineRouteDeclined):  # duplicate angles
        fit_two_component_via_engine(model, c2, duplicate, NLSQConfig())


@pytest.mark.parametrize("deliberate", [True, False])
def test_dispatcher_only_treats_a_deliberate_decline_as_routine(monkeypatch, caplog, deliberate):
    import xpcsjax.optimization.nlsq as nlsq_pkg
    from tests.optimization.test_heterodyne_hybrid_streaming import (
        _Cfg,
        _install_model_stub,
        _make_synthetic_heterodyne,
    )
    from xpcsjax.optimization.exceptions import EngineRouteDeclined

    model, c2, phi = _make_synthetic_heterodyne()
    _install_model_stub(monkeypatch, model)
    error = EngineRouteDeclined("not this fit") if deliberate else NotImplementedError("jax op")

    def engine(*args, **kwargs):
        raise error

    class _Result:
        parameters = np.zeros(1)
        chi_squared = 0.0
        reduced_chi_squared = 0.0
        success = True
        message = "ok"
        nlsq_diagnostics: dict = {}

    monkeypatch.setattr(
        "xpcsjax.optimization.nlsq.heterodyne_engine_route.fit_two_component_via_engine", engine
    )
    monkeypatch.setattr(
        "xpcsjax.optimization.nlsq.heterodyne_core.fit_nlsq_multi_phi", lambda *a, **k: _Result()
    )
    cfg = _Cfg({"analysis_mode": "two_component", "optimization": {"nlsq": {}}})
    with caplog.at_level(logging.INFO):
        nlsq_pkg.fit_nlsq({"c2_exp": c2, "phi_angles_list": phi}, cfg)

    unexpected = [r for r in caplog.records if "failed unexpectedly" in r.getMessage()]
    # A real NotImplementedError from inside the solve must be a visible warning.
    assert bool(unexpected) is (not deliberate)
    if unexpected:
        assert unexpected[0].levelno == logging.WARNING
