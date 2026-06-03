"""Tests for the real joint CMA-ES escape (Task 2).

The escape (:func:`_fit_joint_cmaes_multi_phi`) lifts heterodyne's proven
per-angle CMA-ES pattern to the joint multi-angle objective: NLSQ warm-start →
seed-pinned ``fit_with_cmaes`` over the joint residual → keep-better. It is
additive — the plain joint fit (``enable_cmaes`` off) is unchanged.

These tests pin:

1. A valid escape conserves SSR (``chi2_per_angle.sum() == chi_squared``), is
   keep-better vs the plain fit, and is tagged ``global_escape="cmaes*"``.
2. The seed-pinned escape is bit-reproducible run to run.
3. The escape falls back to the plain joint fit (does NOT raise) when
   ``fit_with_cmaes`` blows up.
"""

from __future__ import annotations

import math

import numpy as np

from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi


def _plain_ssr(model, c2, phi) -> float:
    cfg = NLSQConfig.from_dict(
        {"analysis_mode": "two_component", "per_angle_mode": "averaged"}
    )
    return float(fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None).chi_squared)


def test_joint_cmaes_escape_valid_keep_better_and_tagged():
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    plain = _plain_ssr(model, c2, phi)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "averaged",
            "enable_cmaes": True,
        }
    )
    res = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)
    diag = res.nlsq_diagnostics
    # SSR conservation
    assert np.isclose(
        float(np.sum(diag["chi2_per_angle"])), res.chi_squared, rtol=1e-6
    )
    # keep-better: escape never worse than the plain fit
    assert res.chi_squared <= plain * (1 + 1e-6)
    # the real escape ran (tagged)
    assert diag.get("global_escape", "").startswith("cmaes")


def test_joint_cmaes_escape_deterministic():
    # Fresh model per run: ``HeterodyneModel`` is stateful (the fit mutates
    # ``model.scaling`` / params), and the escape's ``_build_joint_problem``
    # reads the live scaling for its warm-start x0 — so reusing one model across
    # runs would feed run 2 a different (already-mutated) starting point. The
    # determinism contract under test is "same seed → same result for the same
    # inputs", which is what a fresh-model-per-fit (real-usage) pattern exercises.
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "averaged",
            "enable_cmaes": True,
        }
    )
    m1, c1, p1 = make_synthetic_two_component(n_phi=3, n_t=20)
    m2, c2_, p2 = make_synthetic_two_component(n_phi=3, n_t=20)
    r1 = fit_nlsq_multi_phi(m1, c1, p1, cfg, weights=None)
    r2 = fit_nlsq_multi_phi(m2, c2_, p2, cfg, weights=None)
    # Reproducible to numerical precision, not bit-identical: the seed-pinned
    # escape is deterministic, but the warm-start solve's converged values differ
    # by float noise across BLAS backends (macOS Accelerate vs Linux OpenBLAS),
    # so an exact array_equal is platform-fragile while reproducibility is the
    # real contract. The escape branch must still be the same (same tag).
    assert r1.global_escape == r2.global_escape
    assert np.allclose(
        np.asarray(r1.parameters), np.asarray(r2.parameters), rtol=1e-6, atol=1e-8
    )
    assert math.isclose(r1.chi_squared, r2.chi_squared, rel_tol=1e-6, abs_tol=1e-9)


def test_joint_cmaes_escape_falls_back_on_failure(monkeypatch):
    import xpcsjax.optimization.nlsq.heterodyne_core as hc

    def _boom(**k):
        raise RuntimeError("cmaes boom")

    monkeypatch.setattr(hc, "fit_with_cmaes", _boom, raising=False)
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "averaged",
            "enable_cmaes": True,
        }
    )
    # must NOT raise — best-effort fallback to the plain joint fit
    res = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)
    assert res is not None and getattr(res, "parameters", None) is not None


# ---------------------------------------------------------------------------
# Task 3: real joint MULTISTART escape + dispatch wiring.
#
# Heterodyne ``NLSQConfig`` (``heterodyne_config``) exposes multistart as a
# FLAT ``multistart: bool`` + ``multistart_n: int`` pair (NOT a nested dict and
# NOT the ``multi_start_n_starts`` vocabulary of the homodyne config). The
# dispatch gate is ``getattr(config, "multistart", False)``; ``multistart_n``
# drives the LHS start count. The LHS seed is pinned to ``_JOINT_MULTISTART_SEED``
# so the global search is bit-reproducible per fresh model.
# ---------------------------------------------------------------------------
def test_joint_multistart_escape_runs_multiangle_keep_better_tagged():
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    plain = _plain_ssr(model, c2, phi)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "averaged",
            "multistart": True,
            "multistart_n": 3,
        }
    )
    res = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)  # n_phi=3 -> multi-angle
    diag = res.nlsq_diagnostics
    # SSR conservation
    assert np.isclose(
        float(np.sum(diag["chi2_per_angle"])), res.chi_squared, rtol=1e-6
    )
    # keep-better: escape never worse than the plain fit
    assert res.chi_squared <= plain * (1 + 1e-6)
    # the real escape ran (tagged)
    assert diag.get("global_escape", "").startswith("multistart")


def test_joint_multistart_escape_deterministic():
    # Fresh model per run (HeterodyneModel is stateful — see the CMA-ES
    # determinism note above). Seed-pinned LHS ⇒ same inputs → same result.
    cfg_kw = {
        "analysis_mode": "two_component",
        "per_angle_mode": "averaged",
        "multistart": True,
        "multistart_n": 3,
    }

    def run():
        model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
        return fit_nlsq_multi_phi(
            model, c2, phi, NLSQConfig.from_dict(cfg_kw), weights=None
        )

    r1, r2 = run(), run()
    assert np.array_equal(np.asarray(r1.parameters), np.asarray(r2.parameters))


def test_joint_multistart_escape_falls_back_on_failure(monkeypatch):
    import xpcsjax.optimization.nlsq.heterodyne_core as hc

    def _boom(**k):
        raise RuntimeError("ms boom")

    monkeypatch.setattr(hc, "run_multistart_nlsq", _boom, raising=False)
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "averaged",
            "multistart": True,
            "multistart_n": 3,
        }
    )
    # must NOT raise — best-effort fallback to the plain joint fit
    res = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)
    assert res is not None and getattr(res, "parameters", None) is not None


# ---------------------------------------------------------------------------
# Per-angle scaling-mode parity: the global escapes must optimize the SAME
# ``[physics | scaling]`` vector the plain path would. The DEFAULT ``auto``
# mode resolves to ``averaged`` (2 scaling params) for n_phi >= 3; explicit
# ``constant`` freezes scaling (0 scaling params). Enabling CMA-ES / multistart
# must NOT silently switch the layout to Fourier. (Mirrors laminar's CMA-ES,
# which honours ``use_averaged_scaling``.)
# ---------------------------------------------------------------------------
def _plain_diag_and_size(per_angle_mode: str):
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict(
        {"analysis_mode": "two_component", "per_angle_mode": per_angle_mode}
    )
    res = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)
    return res.nlsq_diagnostics, int(np.asarray(res.parameters).size)


def test_joint_cmaes_escape_honors_averaged_default():
    # auto + n_phi>=3 → averaged (2 scaling params). The escape must report the
    # averaged layout, NOT collapse to Fourier.
    plain_diag, plain_size = _plain_diag_and_size("auto")
    assert plain_diag["per_angle_mode"] == "averaged"  # sanity: plain path baseline

    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "auto",
            "enable_cmaes": True,
        }
    )
    res = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)
    diag = res.nlsq_diagnostics
    assert diag.get("global_escape", "").startswith("cmaes")
    assert diag["per_angle_mode"] == "averaged"
    assert diag.get("fourier_basis_dim") is None
    assert int(np.asarray(res.parameters).size) == plain_size


def test_joint_cmaes_escape_honors_constant_explicit():
    # explicit constant → frozen scaling (0 scaling params). The escape is a
    # global search over physics only; scaling stays quantile-frozen.
    plain_diag, plain_size = _plain_diag_and_size("constant")
    assert plain_diag["per_angle_mode"] == "constant"  # sanity: plain path baseline

    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "constant",
            "enable_cmaes": True,
        }
    )
    res = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)
    diag = res.nlsq_diagnostics
    assert diag.get("global_escape", "").startswith("cmaes")
    assert diag["per_angle_mode"] == "constant"
    assert diag.get("fourier_basis_dim") is None
    assert int(np.asarray(res.parameters).size) == plain_size


def test_joint_multistart_escape_honors_averaged_default():
    plain_diag, plain_size = _plain_diag_and_size("auto")
    assert plain_diag["per_angle_mode"] == "averaged"

    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "auto",
            "multistart": True,
            "multistart_n": 3,
        }
    )
    res = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)
    diag = res.nlsq_diagnostics
    assert diag.get("global_escape", "").startswith("multistart")
    assert diag["per_angle_mode"] == "averaged"
    assert diag.get("fourier_basis_dim") is None
    assert int(np.asarray(res.parameters).size) == plain_size
