"""Parity: heterodyne JOINT CMA-ES auto-skip must gate on warm-start SUCCESS.

Defect (RCA 2026-06-16, codex+agy+claude three-way verification): the joint
auto-skip introduced for laminar parity skipped CMA-ES on ``finite SSR/dof <
threshold`` ALONE, with no convergence check. But:

* laminar_flow's ``fit_nlsq_cmaes`` (core.py:2308-2332) sets
  ``nlsq_warmstart_params`` / ``nlsq_warmstart_chi2`` to finite values ONLY when
  ``warmstart_result["success"]`` is True, so its auto-skip gate
  (``params is not None and chi2 < inf``) fires ONLY on a converged warm-start.
* the heterodyne per-angle escape ``_fit_cmaes`` (heterodyne_core.py:3514)
  explicitly checks ``nlsq_result.success`` before skipping.

Because XPCS ``C2`` data is normalized (≈1) the warm-start ``SSR/dof`` is a tiny
MSE (≪ the default threshold 5.0). So a DEGENERATE warm-start that does not
converge but reverts to a low-SSR ``x0`` (exactly the C044 ``two_component``
case the CMA-ES escape exists to rescue) would be auto-skipped — defeating the
escape. These tests pin the success gate: a non-converged warm-start must NOT
auto-skip, regardless of how small its SSR is.
"""

from __future__ import annotations

import numpy as np
import pytest

import xpcsjax.optimization.nlsq.heterodyne_core as hc
from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig

_LB = np.array([-10.0, -10.0])
_UB = np.array([10.0, 10.0])


def _cfg(**overrides):
    base = {
        "analysis_mode": "two_component",
        "per_angle_mode": "averaged",
        "enable_cmaes": True,
    }
    base.update(overrides)
    return NLSQConfig.from_dict(base)


# ---------------------------------------------------------------------------
# Decision function: the success flag is the deciding factor on a good SSR.
# ---------------------------------------------------------------------------
def test_decision_no_skip_when_warmstart_not_converged():
    """SSR/dof = 0 (perfect) but ``warm_success=False`` ⇒ must NOT skip."""
    skip, _reduced = hc._warmstart_auto_skip_decision(
        _cfg(), "cmaes", 0.0, 100, 2, warm_success=False
    )
    assert skip is False


def test_decision_skips_when_warmstart_converged():
    """Control: identical inputs WITH ``warm_success=True`` ⇒ skip (unchanged)."""
    skip, _reduced = hc._warmstart_auto_skip_decision(
        _cfg(), "cmaes", 0.0, 100, 2, warm_success=True
    )
    assert skip is True


# ---------------------------------------------------------------------------
# Shared escape entry (averaged / constant): honors the success flag.
# ---------------------------------------------------------------------------
def test_apply_global_escape_no_skip_when_warmstart_failed(monkeypatch):
    """A low-SSR but NON-converged warm-start still runs the CMA-ES search."""
    called = {"cmaes": False}

    def _spy(_base, x_warm, *_a, **_k):
        called["cmaes"] = True
        return np.asarray(x_warm, dtype=np.float64), True

    monkeypatch.setattr(hc, "_cmaes_joint_candidate", _spy)
    _x, tag, _kept_success = hc._apply_global_escape(
        "cmaes",
        lambda _x: np.zeros(100),
        np.array([1.0, 2.0]),
        _LB,
        _UB,
        _cfg(),
        ["a", "b"],
        _cfg(),
        {},
        warm_success=False,
    )
    assert called["cmaes"] is True
    assert tag != "cmaes_warmstart_auto_skip"


def test_apply_global_escape_still_skips_on_converged_warmstart(monkeypatch):
    """Control: default (``warm_success=True``) still auto-skips on good SSR."""
    called = {"cmaes": False}

    def _spy(*_a, **_k):
        called["cmaes"] = True
        return None, False

    monkeypatch.setattr(hc, "_cmaes_joint_candidate", _spy)
    _x, tag, _kept_success = hc._apply_global_escape(
        "cmaes",
        lambda _x: np.zeros(100),
        np.array([1.0, 2.0]),
        _LB,
        _UB,
        _cfg(),
        ["a", "b"],
        _cfg(),
        {},
    )
    assert tag == "cmaes_warmstart_auto_skip"
    assert called["cmaes"] is False


# ---------------------------------------------------------------------------
# Call-site wiring: each joint escape must EXPLICITLY thread a real bool
# warm-start success flag. A ``_SENTINEL`` default catches a call site that
# forgot to pass it (which a plain ``=True`` default would silently mask). The
# bool's VALUE is fixture-dependent (these degenerate two_component fixtures may
# or may not converge) — the SEMANTICS are pinned by the unit tests above; here
# we pin only that a real bool is threaded through, not its value.
# ---------------------------------------------------------------------------
_SENTINEL = object()


def test_individual_escape_threads_warm_success(monkeypatch):
    """``_fit_joint_cmaes_multi_phi`` passes ``warm.success`` into the decision."""
    captured: dict = {}

    def _capture(config, kind, ssr, n_data, n_params, warm_success=_SENTINEL, **_kw):
        captured["warm_success"] = warm_success
        # Force skip so the test returns immediately without a real CMA-ES run.
        return True, 0.0

    monkeypatch.setattr(hc, "_warmstart_auto_skip_decision", _capture)
    model, c2, phi = make_synthetic_two_component(n_phi=2, n_t=12)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "individual",
            "enable_cmaes": True,
            "cmaes_max_iterations": 5,
            "max_nfev": 30,
        }
    )
    hc.fit_nlsq_multi_phi(model, c2, phi, cfg, None)
    assert captured.get("warm_success", _SENTINEL) is not _SENTINEL
    assert isinstance(captured["warm_success"], bool)


def test_averaged_callsite_threads_warm_success(monkeypatch):
    """``_fit_joint_averaged_multi_phi`` passes the warm result's success."""
    captured: dict = {}

    def _capture(kind, base, x, lb, ub, jc, names, cfg, data, warm_success=_SENTINEL):
        captured["warm_success"] = warm_success
        return np.asarray(x, dtype=np.float64), None, None

    monkeypatch.setattr(hc, "_apply_global_escape", _capture)
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=16)
    cfg = NLSQConfig.from_dict(
        {"analysis_mode": "two_component", "per_angle_mode": "averaged", "enable_cmaes": True}
    )
    hc.fit_nlsq_multi_phi(model, c2, phi, cfg, None)
    assert captured.get("warm_success", _SENTINEL) is not _SENTINEL
    assert isinstance(captured["warm_success"], bool)


# ---------------------------------------------------------------------------
# Covariance preserved on auto-skip (laminar parity: fit_nlsq_cmaes returns
# nlsq_warmstart_cov on skip). The auto-skip keeps the CONVERGED warm-start
# vector unchanged, so its real covariance/uncertainties must NOT be NaN-filled
# (the escape NaN-fill applies only to a vector-MOVING global search).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not hc.HAS_CMAES, reason="cmaes backend not importable")
def test_averaged_auto_skip_preserves_covariance():
    """n_phi=3 averaged warm-start converges ⇒ auto-skip ⇒ finite covariance."""
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "averaged",
            "enable_cmaes": True,
            "cmaes_max_iterations": 5,
        }
    )
    res = hc.fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)
    assert res.nlsq_diagnostics.get("global_escape") == "cmaes_warmstart_auto_skip"
    # Covariance / uncertainties came from the converged warm-start, not NaN.
    assert res.covariance is not None
    assert np.isfinite(np.asarray(res.covariance)).all()
    assert np.isfinite(np.asarray(res.uncertainties)).all()


@pytest.mark.skipif(not hc.HAS_CMAES, reason="cmaes backend not importable")
def test_individual_auto_skip_preserves_covariance():
    """n_phi=2 individual warm-start converges (max_nfev=500) ⇒ finite covariance."""
    model, c2, phi = make_synthetic_two_component(n_phi=2, n_t=12)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "individual",
            "enable_cmaes": True,
            "cmaes_max_iterations": 5,
            "max_nfev": 500,
        }
    )
    res = hc.fit_nlsq_multi_phi(model, c2, phi, cfg, None)
    assert res.nlsq_diagnostics.get("global_escape") == "cmaes_warmstart_auto_skip"
    assert res.covariance is not None
    assert np.isfinite(np.asarray(res.covariance)).all()
    assert np.isfinite(np.asarray(res.uncertainties)).all()


# ---------------------------------------------------------------------------
# Kept-CMAES branch: ``CMAESResult.success`` can be True purely from a
# post-search NLSQ refinement polish even when the global search itself
# exhausted its restart budget without meeting a real convergence criterion
# (cmaes_wrapper.py CR-5, ``success = cmaes_converged or nlsq_refined``). A
# kept vector with SSR <= warm SSR under those conditions must NOT report
# converged/good — see 4f0a35c ("implement CMA-ES warm-start auto-skip with
# success gate for joint fits") and ad90201 ("report failure on keep-better
# floor-reverted warm-start to prevent spurious success") for the two prior
# rounds of this exact bug class (auto-skip gate, floor-revert gate); this
# is the third: the "kept cmaes" branch itself.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not hc.HAS_CMAES, reason="cmaes backend not importable")
def test_kept_cmaes_refinement_only_success_reports_marginal(monkeypatch):
    """``escape='cmaes'`` kept purely via ``nlsq_refined`` ⇒ marginal/failed, not good/converged."""
    from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESResult

    def _fake_fit_with_cmaes(model_func, xdata, ydata, p0, bounds, sigma=None, config=None):
        # Return the warm-start vector unchanged (SSR tie ⇒ satisfies the
        # keep-better comparison) tagged exactly like a real max_restarts
        # global search rescued only by the refinement polish.
        return CMAESResult(
            parameters=np.asarray(p0, dtype=np.float64),
            covariance=None,
            chi_squared=0.0,
            success=True,
            diagnostics={"convergence_reason": "max_restarts"},
            method_used="cmaes",
            nlsq_refined=True,
        )

    monkeypatch.setattr(hc, "fit_with_cmaes", _fake_fit_with_cmaes)
    # Force Phase 2 to run regardless of warm-start SSR/dof (mirrors the other
    # call-site tests above) so this test isolates the kept-branch logic.
    monkeypatch.setattr(hc, "_warmstart_auto_skip_decision", lambda *a, **k: (False, 0.0))

    model, c2, phi = make_synthetic_two_component(n_phi=2, n_t=12)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "individual",
            "enable_cmaes": True,
            "cmaes_max_iterations": 5,
            "max_nfev": 30,
        }
    )
    res = hc._fit_joint_cmaes_multi_phi(model, c2, phi, cfg, None)
    assert res.nlsq_diagnostics.get("global_escape") == "cmaes"
    assert res.quality_flag == "marginal"
    assert res.convergence_status == "failed"


@pytest.mark.skipif(not hc.HAS_CMAES, reason="cmaes backend not importable")
@pytest.mark.parametrize("nlsq_refined", [False, True])
def test_kept_cmaes_real_convergence_still_reports_good(monkeypatch, nlsq_refined):
    """Control: ``escape='cmaes'`` kept via a REAL convergence reason ⇒ still good/converged.

    ``"xtol"`` is the only reason NLSQ's ``CMAESOptimizer`` reports for actual
    convergence (verified against the pinned nlsq backend — see the
    ``CMAES_CONVERGED_REASONS`` docstring in cmaes_wrapper.py). Parametrized
    over ``nlsq_refined`` so a genuinely-converged search that ALSO got a
    refinement polish (the common real-world case) is pinned too, not just
    the unrefined case.
    """
    from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESResult

    def _fake_fit_with_cmaes(model_func, xdata, ydata, p0, bounds, sigma=None, config=None):
        return CMAESResult(
            parameters=np.asarray(p0, dtype=np.float64),
            covariance=None,
            chi_squared=0.0,
            success=True,
            diagnostics={"convergence_reason": "xtol"},
            method_used="cmaes",
            nlsq_refined=nlsq_refined,
        )

    monkeypatch.setattr(hc, "fit_with_cmaes", _fake_fit_with_cmaes)
    monkeypatch.setattr(hc, "_warmstart_auto_skip_decision", lambda *a, **k: (False, 0.0))

    model, c2, phi = make_synthetic_two_component(n_phi=2, n_t=12)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "individual",
            "enable_cmaes": True,
            "cmaes_max_iterations": 5,
            "max_nfev": 30,
        }
    )
    res = hc._fit_joint_cmaes_multi_phi(model, c2, phi, cfg, None)
    assert res.nlsq_diagnostics.get("global_escape") == "cmaes"
    assert res.quality_flag == "good"
    assert res.convergence_status == "converged"


def test_constant_callsite_threads_warm_success(monkeypatch):
    """``_fit_joint_constant_multi_phi`` passes ``nlsq_result.success``."""
    captured: dict = {}

    def _capture(kind, base, x, lb, ub, jc, names, cfg, data, warm_success=_SENTINEL):
        captured["warm_success"] = warm_success
        return np.asarray(x, dtype=np.float64), None, None

    monkeypatch.setattr(hc, "_apply_global_escape", _capture)
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=16)
    cfg = NLSQConfig.from_dict(
        {"analysis_mode": "two_component", "per_angle_mode": "constant", "enable_cmaes": True}
    )
    hc.fit_nlsq_multi_phi(model, c2, phi, cfg, None)
    assert captured.get("warm_success", _SENTINEL) is not _SENTINEL
    assert isinstance(captured["warm_success"], bool)
