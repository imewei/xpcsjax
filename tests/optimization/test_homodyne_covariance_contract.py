"""Homodyne covariance contract: no fabricated uncertainties on any wrapper path.

F1: a strategy that could not produce a real covariance sets
``covariance_is_placeholder`` and the wrapper honours it (NaN sigma, flag in
``nlsq_diagnostics``) instead of shipping ``sqrt(diag(I)) = 1.0``.
F2: no ``1e-5`` floor — a singular / null-space / non-finite variance is NaN.
F3: the early OUT_OF_CORE branch expands the compact ``[c, o | physics]`` x0 to
the per-angle layout the kernel slices, so the fit is not corrupted.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.optimization.test_fixed_parameters_integration import (
    _ALL_PHYSICAL_NAMES,
    CONTRAST,
    OFFSET,
    _config,
    _synthetic_data,
)
from xpcsjax.config import ConfigManager
from xpcsjax.optimization.nlsq.core import fit_nlsq_jax
from xpcsjax.optimization.nlsq.recovery import safe_uncertainties_from_pcov
from xpcsjax.optimization.nlsq.result_helpers import _uncertainties_from_pcov


def test_recovery_helper_has_no_floor():
    diag = np.array([np.inf, np.nan, 0.0, -1.0, 4.0])
    out = safe_uncertainties_from_pcov(np.diag(diag), 5)
    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isnan(out[3])
    assert out[2] == 0.0 and out[4] == 2.0
    assert np.all(np.isnan(safe_uncertainties_from_pcov(np.eye(3), 5)))


def test_wrapper_helper_honours_placeholder_and_structural_zero():
    cov, unc = _uncertainties_from_pcov(np.diag([1.0, 0.0, 4.0]), 3)
    np.testing.assert_array_equal(unc, [1.0, 0.0, 2.0])
    cov, unc = _uncertainties_from_pcov(np.eye(3), 3, is_placeholder=True)
    assert np.all(np.isnan(cov)) and np.all(np.isnan(unc))
    cov, unc = _uncertainties_from_pcov(None, 2)
    assert np.all(np.isnan(unc))


def _force_strategy(monkeypatch, strategy_name: str):
    from xpcsjax.optimization.nlsq import wrapper as wrapper_module
    from xpcsjax.optimization.nlsq.memory import NLSQStrategy, StrategyDecision

    def _force(n_points, n_params, *args, **kwargs):
        return StrategyDecision(
            strategy=getattr(NLSQStrategy, strategy_name),
            threshold_gb=0.001,
            index_memory_gb=0.0,
            peak_memory_gb=1.0,
            reason=f"forced {strategy_name}",
        )

    monkeypatch.setattr(wrapper_module, "select_nlsq_strategy", _force)


def test_early_out_of_core_branch_expands_compact_x0(monkeypatch):
    """F3: chi2 was ~1e13 with a negative covariance diagonal when the kernel
    sliced the compact vector as per-angle scaling."""
    _force_strategy(monkeypatch, "OUT_OF_CORE")
    data = _synthetic_data("laminar_flow")
    n_phi = int(np.unique(np.asarray(data["phi"])).size)
    cm = ConfigManager(config_override=_config("laminar_flow"))
    res = fit_nlsq_jax(data, cm, use_adapter=False)
    assert res.recovery_actions == ["out_of_core_delegation"]
    params = np.asarray(res.parameters).ravel()
    assert params.size == 2 * n_phi + len(_ALL_PHYSICAL_NAMES)
    # Per-angle contrast / offset land on the simulated values.
    np.testing.assert_allclose(params[:n_phi], CONTRAST, rtol=0.1)
    np.testing.assert_allclose(params[n_phi : 2 * n_phi], OFFSET, rtol=0.1)
    assert np.isfinite(res.chi_squared) and res.chi_squared < 1e3
    unc = np.asarray(res.uncertainties)
    fin = np.isfinite(unc)
    assert np.all(unc[fin] >= 0.0)
    assert "covariance_is_placeholder" in res.nlsq_diagnostics


@pytest.mark.parametrize("fixed", [None, {"D_offset": 37.5}])
def test_out_of_core_singular_hessian_reports_nan_and_flag(monkeypatch, fixed):
    """F1/F2 on out-of-core: a singular free-block JᵀJ is NaN + flag, not pinv/1e-5.

    The seam is ``covariance.finalize_covariance`` as imported by out_of_core
    (the only way its free-block verdict is produced), forced to the
    placeholder verdict. With a fixed physical parameter the fixed slot still
    reads exactly 0.0 (known by construction); every free slot is NaN.
    """
    import xpcsjax.optimization.nlsq.strategies.out_of_core as ooc

    _force_strategy(monkeypatch, "OUT_OF_CORE")

    def _always_placeholder(pcov, n):
        return np.full((n, n), np.nan), np.full(n, np.nan), True

    monkeypatch.setattr(ooc, "finalize_covariance", _always_placeholder)
    data = _synthetic_data("laminar_flow")
    cm = ConfigManager(config_override=_config("laminar_flow", fixed_parameters=fixed))
    res = fit_nlsq_jax(data, cm, use_adapter=False)
    assert res.recovery_actions == ["out_of_core_delegation"]
    assert res.nlsq_diagnostics["covariance_is_placeholder"] is True
    unc = np.asarray(res.uncertainties)
    if fixed:
        i = len(unc) - len(_ALL_PHYSICAL_NAMES) + _ALL_PHYSICAL_NAMES.index("D_offset")
        assert unc[i] == 0.0
        unc = np.delete(unc, i)
    assert np.all(np.isnan(unc))


@pytest.mark.filterwarnings("ignore:Ill-conditioned Jacobian.*:UserWarning:nlsq.stability.guard")
def test_laminar_stratified_ls_never_ships_identity_sigma():
    """F1 on the >=1M stratified-LS path (direct strategy call): whenever the
    strategy has no real Gauss-Newton covariance (accepted L2 popt, or a
    singular JᵀJ) it reports NaN + flag, never ``J_full @ I @ J_fullᵀ`` whose
    diagonal is exactly 1.0. Both outcomes are asserted, so the test is never
    vacuous: a real covariance must be finite with positive free variances."""
    from tests.parity.test_laminar_execute_layers import _fit

    popt, info = _fit(execute_layers=True)
    ad = info.get("anti_degeneracy", {})
    flag = bool(info.get("covariance_is_placeholder", False) or ad.get("covariance_is_placeholder"))
    if ad.get("execute_layers_kind") == "L2_hierarchical":
        assert flag is True, "accepted L2 has no GN covariance -> must be flagged"
    else:
        assert flag is False
    # The strategy's own pcov: on a placeholder it is all-NaN (no identity).
    _popt2, pcov2, info2 = _fit_with_pcov(execute_layers=True)
    if info2.get("covariance_is_placeholder"):
        assert np.all(np.isnan(pcov2))
    else:
        assert np.all(np.isfinite(pcov2))
        assert not np.allclose(np.diag(pcov2), 1.0)


def _fit_with_pcov(execute_layers: bool):
    """Same as tests.parity.test_laminar_execute_layers._fit but returns pcov too."""
    from tests.parity.test_laminar_execute_layers import (
        _LOG,
        _NAMES,
        _build_laminar_stratified_data,
    )
    from xpcsjax.config.parameter_registry import AnalysisMode
    from xpcsjax.optimization.nlsq.strategies.stratified_ls import (
        fit_with_stratified_least_squares,
    )

    strat, init, bounds, _ = _build_laminar_stratified_data()
    ad = {
        "enable": True,
        "per_angle_mode": "individual",
        "execute_layers": execute_layers,
        "hierarchical": {
            "enable": True,
            "max_outer_iterations": 2,
            "physical_max_iterations": 30,
            "per_angle_max_iterations": 20,
        },
        "regularization": {"enable": True, "mode": "relative", "lambda": 1.0},
    }
    return fit_with_stratified_least_squares(
        strat,
        True,
        _NAMES,
        init,
        bounds,
        _LOG,
        target_chunk_size=2000,
        anti_degeneracy_config=ad,
        nlsq_config_dict={"max_iterations": 100},
        analysis_mode=AnalysisMode.LAMINAR_FLOW,
    )


def test_failed_fit_sentinels_are_nan_and_flagged():
    from xpcsjax.optimization.nlsq.core import _cmaes_failed_result
    from xpcsjax.optimization.nlsq.multistart import MultiStartResult, SingleStartResult

    r = _cmaes_failed_result(np.array([1.0, 2.0, 3.0]), 0.0, RuntimeError("boom"))
    assert np.all(np.isnan(r.uncertainties)) and np.all(np.isnan(r.covariance))
    assert r.nlsq_diagnostics["covariance_is_placeholder"] is True

    best = SingleStartResult(
        start_idx=0,
        initial_params=np.zeros(2),
        final_params=np.ones(2),
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        success=True,
        covariance=None,
    )
    ms = MultiStartResult(all_results=[best], best=best, config=None, strategy_used="lhs")
    res = ms.to_optimization_result()
    assert np.all(np.isnan(res.uncertainties)) and np.all(np.isnan(res.covariance))
    assert res.nlsq_diagnostics["covariance_is_placeholder"] is True
    ok = SingleStartResult(
        start_idx=0,
        initial_params=np.zeros(2),
        final_params=np.ones(2),
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        success=True,
        covariance=np.diag([4.0, 9.0]),
    )
    res_ok = MultiStartResult(
        all_results=[ok], best=ok, config=None, strategy_used="lhs"
    ).to_optimization_result()
    np.testing.assert_array_equal(res_ok.uncertainties, [2.0, 3.0])
    assert res_ok.nlsq_diagnostics["covariance_is_placeholder"] is False


@pytest.mark.filterwarnings("ignore:Ill-conditioned Jacobian.*:UserWarning:nlsq.stability.guard")
def test_sequential_singular_angle_is_excluded_not_weighted_as_sigma_one():
    """A per-angle singular JᵀJ yields a NaN per-angle covariance that the
    inverse-variance combination EXCLUDES; a parameter singular at every angle
    combines to NaN (unknown), never to a fabricated 1/(1+...)."""
    import jax.numpy as jnp

    from xpcsjax.optimization.nlsq.strategies.sequential import (
        combine_angle_results,
        optimize_per_angle_sequential,
    )

    def residual(params, phi, t1, t2, g2):
        # params[1] never enters the model: its J column is zero at every angle.
        return jnp.asarray(g2 - (params[0] + 0.0 * params[1]))

    n = 12
    phi = np.concatenate([np.zeros(n), np.full(n, 45.0)])
    t1 = np.tile(np.linspace(0.0, 1.0, n), 2)
    t2 = np.zeros(2 * n)
    g2 = 2.0 + np.concatenate([0.01 * np.sin(np.arange(n)), 0.01 * np.cos(np.arange(n))])
    res = optimize_per_angle_sequential(
        phi=phi,
        t1=t1,
        t2=t2,
        g2_exp=g2,
        residual_func=residual,
        initial_params=np.array([1.0, 1.0]),
        bounds=(np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
    )
    var = np.diag(res.combined_covariance)
    assert np.isfinite(var[0]) and var[0] > 0
    assert np.isnan(var[1]), var
    # Direct combination check: a fully-NaN angle is excluded, not weighted.
    good = {
        "parameters": np.array([1.0, 2.0]),
        "covariance": np.diag([1.0, 4.0]),
        "cost": 1.0,
        "success": True,
        "n_points": 10,
    }
    bad = dict(good, covariance=np.full((2, 2), np.nan), parameters=np.array([100.0, 100.0]))
    p, c, _ = combine_angle_results([good, bad], weighting="inverse_variance")
    np.testing.assert_allclose(p, [1.0, 2.0])
    np.testing.assert_allclose(np.diag(c), [1.0, 4.0])


def test_wrapper_create_fit_result_flags_placeholder():
    from xpcsjax.optimization.nlsq.wrapper import NLSQWrapper

    w = NLSQWrapper.__new__(NLSQWrapper)
    res = NLSQWrapper._create_fit_result(
        w,
        popt=np.array([1.0, 2.0]),
        pcov=np.eye(2),
        residuals=np.zeros(10),
        n_data=10,
        iterations=1,
        execution_time=0.0,
        covariance_is_placeholder=True,
    )
    assert np.all(np.isnan(res.uncertainties)) and np.all(np.isnan(res.covariance))
    assert res.nlsq_diagnostics["covariance_is_placeholder"] is True
    ok = NLSQWrapper._create_fit_result(
        w,
        popt=np.array([1.0, 2.0]),
        pcov=np.diag([4.0, 0.0]),
        residuals=np.zeros(10),
        n_data=10,
        iterations=1,
        execution_time=0.0,
    )
    np.testing.assert_array_equal(ok.uncertainties, [2.0, 0.0])
    assert ok.nlsq_diagnostics["covariance_is_placeholder"] is False
