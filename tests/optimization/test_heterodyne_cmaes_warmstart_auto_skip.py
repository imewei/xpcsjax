"""Parity: heterodyne JOINT CMA-ES escapes honor ``cmaes_warmstart_auto_skip``.

Defect B (RCA 2026-06-15): laminar_flow's ``fit_nlsq_cmaes`` (core.py:2354-2382)
skips the expensive CMA-ES global search when the NLSQ warm-start already lands a
good fit (sigma-normalized reduced χ² = SSR/dof below
``cmaes_warmstart_skip_threshold``). The heterodyne *per-angle* escape
(``_fit_cmaes``, heterodyne_core.py:3431) honors the same knob, but the heterodyne
*joint* escapes did NOT:

* ``_apply_global_escape`` (averaged / constant) called ``_cmaes_joint_candidate``
  unconditionally, and
* ``_fit_joint_cmaes_multi_phi`` (individual) always ran Phase-2 CMA-ES.

So a multi-angle ``two_component`` fit with ``enable_cmaes`` always paid the
multi-minute global search even when the warm-start was already excellent — the
behavioral divergence from laminar (which finishes with ``Generations: 0``).

These tests pin the parity: the joint escapes auto-skip CMA-ES on a good warm
start (tagging ``global_escape="cmaes_warmstart_auto_skip"`` and NOT invoking the
global search), still run it on a poor warm start, honor the disable knob, and
never auto-skip a ``multistart`` escape (the knob is CMA-ES-specific, matching
laminar + the knob name).
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
# Unit-level: the shared escape entry ``_apply_global_escape`` (averaged/constant)
# ---------------------------------------------------------------------------
def test_apply_global_escape_auto_skips_cmaes_on_good_warmstart(monkeypatch):
    """SSR/dof below threshold ⇒ skip: tag the auto-skip and never call CMA-ES."""
    called = {"cmaes": False}

    def _spy(*_a, **_k):
        called["cmaes"] = True
        return None, False

    monkeypatch.setattr(hc, "_cmaes_joint_candidate", _spy)
    x_warm = np.array([1.0, 2.0])
    # 100 residual rows all zero ⇒ SSR=0, dof = 100 - 2 = 98 ⇒ reduced χ² = 0 < 5.
    x_final, tag, _kept_success = hc._apply_global_escape(
        "cmaes", lambda _x: np.zeros(100), x_warm, _LB, _UB, _cfg(), ["a", "b"], _cfg(), {}
    )

    assert tag == "cmaes_warmstart_auto_skip"
    assert np.array_equal(x_final, x_warm)
    assert called["cmaes"] is False


def test_apply_global_escape_runs_cmaes_on_poor_warmstart(monkeypatch):
    """SSR/dof above threshold ⇒ no skip: the global search still runs."""
    called = {"cmaes": False}

    def _spy(_base, x_warm, *_a, **_k):
        called["cmaes"] = True
        return np.asarray(x_warm, dtype=np.float64), True

    monkeypatch.setattr(hc, "_cmaes_joint_candidate", _spy)
    x_warm = np.array([1.0, 2.0])
    # residual 10 everywhere ⇒ SSR = 100*100 = 1e4, dof = 98 ⇒ reduced ≈ 102 ≥ 5.
    _x_final, tag, _kept_success = hc._apply_global_escape(
        "cmaes", lambda _x: np.full(100, 10.0), x_warm, _LB, _UB, _cfg(), ["a", "b"], _cfg(), {}
    )

    assert called["cmaes"] is True
    assert tag != "cmaes_warmstart_auto_skip"


def test_apply_global_escape_respects_disabled_autoskip(monkeypatch):
    """A good warm start does NOT skip when ``cmaes_warmstart_auto_skip=False``."""
    called = {"cmaes": False}

    def _spy(_base, x_warm, *_a, **_k):
        called["cmaes"] = True
        return np.asarray(x_warm, dtype=np.float64), True

    monkeypatch.setattr(hc, "_cmaes_joint_candidate", _spy)
    cfg = _cfg(cmaes_warmstart_auto_skip=False)
    _x_final, tag, _kept_success = hc._apply_global_escape(
        "cmaes", lambda _x: np.zeros(100), np.array([1.0, 2.0]), _LB, _UB, cfg, ["a", "b"], cfg, {}
    )

    assert called["cmaes"] is True
    assert tag != "cmaes_warmstart_auto_skip"


def test_apply_global_escape_never_auto_skips_multistart(monkeypatch):
    """The auto-skip knob is CMA-ES-specific (matches laminar + the knob name)."""
    called = {"ms": False}

    def _spy(_base, x_warm, *_a, **_k):
        called["ms"] = True
        return np.asarray(x_warm, dtype=np.float64)

    monkeypatch.setattr(hc, "_multistart_joint_candidate", _spy)
    _x_final, tag, _kept_success = hc._apply_global_escape(
        "multistart",
        lambda _x: np.zeros(100),
        np.array([1.0, 2.0]),
        _LB,
        _UB,
        _cfg(),
        ["a", "b"],
        _cfg(),
        {},
    )

    assert called["ms"] is True
    assert tag != "cmaes_warmstart_auto_skip"


def test_apply_global_escape_no_skip_when_dof_nonpositive(monkeypatch):
    """dof ≤ 0 (more params than data) never skips — guards a meaningless χ²/dof."""
    called = {"cmaes": False}

    def _spy(_base, x_warm, *_a, **_k):
        called["cmaes"] = True
        return np.asarray(x_warm, dtype=np.float64), True

    monkeypatch.setattr(hc, "_cmaes_joint_candidate", _spy)
    x_warm = np.zeros(100)  # 100 params
    lb = np.full(100, -10.0)
    ub = np.full(100, 10.0)
    # n_data = 50 < n_params = 100 ⇒ dof = -50 ≤ 0, even though SSR = 0.
    _x_final, tag, _kept_success = hc._apply_global_escape(
        "cmaes", lambda _x: np.zeros(50), x_warm, lb, ub, _cfg(), ["p"] * 100, _cfg(), {}
    )

    assert called["cmaes"] is True
    assert tag != "cmaes_warmstart_auto_skip"


# ---------------------------------------------------------------------------
# Integration: end-to-end via ``fit_nlsq_multi_phi`` (the user's real path).
# The synthetic fixture generates data AT the model's initial params (+5e-4
# noise), so the NLSQ warm-start fits near-perfectly ⇒ reduced χ² ≪ 5 ⇒ skip.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not hc.HAS_CMAES, reason="cmaes backend not importable")
def test_averaged_cmaes_escape_auto_skips_on_good_warmstart():
    """n_phi=3 → averaged escape (the user's 3-angle case) auto-skips CMA-ES."""
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


@pytest.mark.skipif(not hc.HAS_CMAES, reason="cmaes backend not importable")
def test_individual_cmaes_escape_auto_skips_on_good_warmstart():
    """n_phi=2 → individual escape (``_fit_joint_cmaes_multi_phi``) auto-skips."""
    model, c2, phi = make_synthetic_two_component(n_phi=2, n_t=12)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "individual",
            "enable_cmaes": True,
            "cmaes_max_iterations": 5,
            # max_nfev must be large enough for the warm-start to actually
            # CONVERGE (success=True) — the auto-skip now gates on convergence
            # (parity with laminar + per-angle _fit_cmaes). At max_nfev=30 the
            # trust-region hits the eval cap before gtol (degenerate two_component
            # Jacobian) → success=False → it (correctly) does NOT auto-skip.
            "max_nfev": 500,
        }
    )
    res = hc.fit_nlsq_multi_phi(model, c2, phi, cfg, None)
    assert res.nlsq_diagnostics.get("global_escape") == "cmaes_warmstart_auto_skip"
