"""One failed-covariance semantics on every heterodyne path.

A covariance that is not a measured estimate — nlsq's all-``inf`` singular
marker, a pseudo-inverse whose null-space directions read as exactly ``0.0``
variance, a missing ``pcov`` — must surface as all-NaN uncertainties with
``covariance_is_placeholder=True``; never ``inf``, never ``0.0``, never an
identity shipped as real.
"""

from __future__ import annotations

import numpy as np
import pytest

import xpcsjax.optimization.nlsq.heterodyne_adapter as had
import xpcsjax.optimization.nlsq.strategies.heterodyne_hybrid_streaming as hs
from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi
from xpcsjax.optimization.nlsq.heterodyne_engine_route import fit_two_component_via_engine
from xpcsjax.optimization.nlsq.heterodyne_result_builder import build_result_from_nlsq
from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
    build_heterodyne_stratified_data,
)


def _cfg(mode: str) -> NLSQConfig:
    c = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": mode})
    return c


def _singular_curve_fit(monkeypatch):
    """Make every nlsq solve report the singular-Jacobian all-inf pcov."""
    _Orig = had.CurveFit

    class _SingularCurveFit(_Orig):
        def curve_fit(self, *a, **k):
            r = super().curve_fit(*a, **k)
            popt = np.asarray(r[0] if isinstance(r, tuple) else r.popt)
            return popt, np.full((popt.size, popt.size), np.inf)

    monkeypatch.setattr(had, "CurveFit", _SingularCurveFit)


def test_build_result_from_nlsq_singular_marker_is_nan_not_inf():
    popt = np.array([1.0, 2.0])
    res = build_result_from_nlsq(
        nlsq_result=(popt, np.full((2, 2), np.inf)), parameter_names=["a", "b"], n_data=10
    )
    assert np.all(np.isnan(res.uncertainties))
    assert np.all(np.isnan(res.covariance))
    assert res.metadata["covariance_is_placeholder"] is True
    # pinv-style exact-zero variance is rejected the same way.
    res0 = build_result_from_nlsq(
        nlsq_result=(popt, np.diag([1.0, 0.0])), parameter_names=["a", "b"], n_data=10
    )
    assert np.all(np.isnan(res0.uncertainties))
    assert res0.metadata["covariance_is_placeholder"] is True
    # An ABSENT solver covariance is the same case, never ``None``.
    res_none = build_result_from_nlsq(nlsq_result=(popt, None), parameter_names=["a", "b"], n_data=10)
    assert res_none.covariance is not None and np.all(np.isnan(res_none.covariance))
    assert np.all(np.isnan(res_none.uncertainties))
    assert res_none.metadata["covariance_is_placeholder"] is True
    # A real covariance is passed through untouched.
    ok = build_result_from_nlsq(
        nlsq_result=(popt, np.diag([1.0, 4.0])), parameter_names=["a", "b"], n_data=10
    )
    np.testing.assert_allclose(ok.uncertainties, [1.0, 2.0])
    assert ok.metadata["covariance_is_placeholder"] is False


@pytest.mark.parametrize("mode", ["constant", "averaged", "individual"])
def test_in_memory_multi_phi_singular_reports_nan_and_flag(monkeypatch, mode):
    _singular_curve_fit(monkeypatch)
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=12)
    res = fit_nlsq_multi_phi(model, c2, list(phi), _cfg(mode), None)
    assert np.all(np.isnan(res.uncertainties)), res.uncertainties
    assert np.all(np.isnan(res.covariance))
    assert res.nlsq_diagnostics["covariance_is_placeholder"] is True
    assert "global_escape" not in res.nlsq_diagnostics


@pytest.mark.parametrize("mode", ["constant", "averaged", "individual"])
def test_engine_route_singular_reports_nan_and_flag(monkeypatch, mode):
    _singular_curve_fit(monkeypatch)
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=12)
    res = fit_two_component_via_engine(model, c2, np.asarray(phi), _cfg(mode), None)
    assert np.all(np.isnan(res.uncertainties))
    assert res.nlsq_diagnostics["covariance_is_placeholder"] is True


@pytest.mark.parametrize("mode", ["constant", "averaged", "individual"])
def test_in_memory_paths_flag_false_on_real_covariance(mode):
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=12)
    for res in (
        fit_nlsq_multi_phi(model, c2, list(phi), _cfg(mode), None),
        fit_two_component_via_engine(model, c2, np.asarray(phi), _cfg(mode), None),
    ):
        flag = res.nlsq_diagnostics["covariance_is_placeholder"]
        finite = bool(np.all(np.isfinite(res.uncertainties)))
        assert flag is (not finite)
        if finite:
            assert np.all(res.uncertainties > 0)


@pytest.mark.parametrize(
    ("pcov_factory", "label"),
    [
        (lambda n: None, "absent"),
        (lambda n: np.diag([0.0] + [1.0] * (n - 1)), "pinv_zero_variance"),
        (lambda n: np.full((n, n), np.inf), "singular_inf"),
    ],
)
def test_streaming_plain_branch_rejects_non_real_pcov(monkeypatch, pcov_factory, label):
    class _FakeOpt:
        def __init__(self, config):
            pass

        def fit(self, data_source, func, p0, bounds=None, sigma=None, **kw):
            n = len(p0)
            out = {"x": np.asarray(p0, dtype=float), "nit": 3, "success": True}
            pc = pcov_factory(n)
            if pc is not None:
                out["pcov"] = pc
            return out

    monkeypatch.setattr(hs, "AdaptiveHybridStreamingOptimizer", _FakeOpt)
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=12)
    strat = build_heterodyne_stratified_data(model, c2, phi, weights=None)
    lower, upper = model.param_manager.get_bounds()
    popt, pcov, info = hs.fit_with_stratified_hybrid_streaming_heterodyne(
        stratified_data=strat,
        model=model,
        physical_param_names=list(model.param_manager.varying_names),
        initial_params=np.asarray(model.param_manager.get_initial_values()),
        bounds=(np.asarray(lower), np.asarray(upper)),
        hybrid_config={"enable": True, "warmup_iterations": 5, "chunk_size": 4000},
        anti_degeneracy_config={"per_angle_mode": "constant"},
    )
    assert pcov.shape == (popt.size, popt.size)
    assert np.all(np.isnan(pcov)), label
    assert info["covariance_is_placeholder"] is True


def test_streaming_plain_branch_keeps_real_pcov(monkeypatch):
    class _FakeOpt:
        def __init__(self, config):
            pass

        def fit(self, data_source, func, p0, bounds=None, sigma=None, **kw):
            n = len(p0)
            return {"x": np.asarray(p0, dtype=float), "pcov": np.eye(n) * 2.0, "nit": 3}

    monkeypatch.setattr(hs, "AdaptiveHybridStreamingOptimizer", _FakeOpt)
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=12)
    strat = build_heterodyne_stratified_data(model, c2, phi, weights=None)
    lower, upper = model.param_manager.get_bounds()
    popt, pcov, info = hs.fit_with_stratified_hybrid_streaming_heterodyne(
        stratified_data=strat,
        model=model,
        physical_param_names=list(model.param_manager.varying_names),
        initial_params=np.asarray(model.param_manager.get_initial_values()),
        bounds=(np.asarray(lower), np.asarray(upper)),
        hybrid_config={"enable": True, "warmup_iterations": 5, "chunk_size": 4000},
        anti_degeneracy_config={"per_angle_mode": "constant"},
    )
    np.testing.assert_array_equal(pcov, np.eye(popt.size) * 2.0)
    assert info["covariance_is_placeholder"] is False


@pytest.mark.parametrize("mode", ["constant", "individual"])
def test_fixed_or_tied_slots_do_not_trip_the_placeholder_flag(tmp_path, mode):
    """The flag judges the SOLVER's reduced covariance, not the expanded one.

    A tied physics slot is not in the optimizer vector; ``expand_reduced_result``
    mirrors its parent's uncertainty into the full-14 layout after the solver
    covariance was finalized. Structural (non-solver) slots must therefore never
    flip the flag, and a real solve keeps ``covariance_is_placeholder=False``.
    """
    from tests.optimization.test_heterodyne_tied_result_assembly import _run_tied_fit

    result = _run_tied_fit(tmp_path, np.array([0.0, 45.0, 90.0]), mode)
    diag = result.nlsq_diagnostics or {}
    assert "tied_parameters" in diag
    unc = np.asarray(result.uncertainties, dtype=np.float64)
    assert diag["covariance_is_placeholder"] is False
    assert np.all(np.isfinite(unc)) and np.all(unc > 0)
