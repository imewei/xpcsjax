"""Red tests for Codex review round 1, shard opthet (claims opthet-B1..B13)."""

from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np

import xpcsjax.optimization.nlsq.heterodyne_core as heterodyne_core
from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq.heterodyne_adapter import NLSQAdapter
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import _build_joint_problem, fit_nlsq_multi_phi
from xpcsjax.optimization.nlsq.heterodyne_engine_route import fit_two_component_via_engine
from xpcsjax.optimization.nlsq.heterodyne_result_builder import build_hybrid_streaming_result
from xpcsjax.optimization.nlsq.heterodyne_results import NLSQResult
from xpcsjax.optimization.nlsq.heterodyne_stratified_data import build_heterodyne_stratified_data
from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
    build_joint_pointwise_residual,
    fit_heterodyne_stratified_least_squares,
)


def _unsorted_two_angle():
    """Noise-free data on phi=[75, 0] with per-angle offsets [1.1, 0.9] (input order)."""
    m, _c2, _phi = make_synthetic_two_component(n_phi=2, n_t=12)
    phi = np.array([75.0, 0.0])
    c2 = np.stack(
        [
            np.asarray(m.compute_correlation(phi_angle=p, angle_idx=i, contrast=0.3, offset=o))
            for i, (p, o) in enumerate(zip(phi, [1.1, 0.9], strict=True))
        ]
    )
    return m, c2, phi


def _patch_exact_fit(monkeypatch, seen):
    """Solver stub (the solver is not under test): return the seed physics with the
    exact scaling head in SORTED-angle order, recording the real residual SSR there."""

    def exact_fit(self, *, residual_fn, initial_params, **kwargs):
        p = np.array(initial_params, dtype=np.float64, copy=True)
        p[:4] = [0.3, 0.3, 0.9, 1.1]
        seen["ssr"] = float(np.sum(np.asarray(residual_fn(p)) ** 2))
        return NLSQResult(p, self._parameter_names, True, "ok")

    monkeypatch.setattr(NLSQAdapter, "fit", exact_fit)


def test_opthet_b1(monkeypatch):
    m, c2, phi = _unsorted_two_angle()
    seen = {}
    _patch_exact_fit(monkeypatch, seen)
    r = fit_two_component_via_engine(m, c2, phi, NLSQConfig(per_angle_mode="individual"))
    assert seen["ssr"] < 1e-20, seen  # premise: this IS the engine's exact optimum
    assert r.chi_squared < 1e-12, (seen, r.chi_squared)


def test_opthet_b2(monkeypatch):
    m, c2, phi = _unsorted_two_angle()
    seen = {}
    _patch_exact_fit(monkeypatch, seen)
    r = fit_heterodyne_stratified_least_squares(
        model=m,
        c2=c2,
        phi=phi,
        weights=None,
        config=NLSQConfig(per_angle_mode="individual"),
    )
    assert seen["ssr"] < 1e-20, seen
    np.testing.assert_allclose(r.nlsq_diagnostics["phi_angles"], [75.0, 0.0])
    np.testing.assert_allclose(r.parameters[2:4], [1.1, 0.9])


def test_opthet_b3():
    model, _c2, phi = make_synthetic_two_component(n_phi=2, n_t=12)
    c2 = np.stack(
        [
            np.asarray(
                model.compute_correlation(phi_angle=p, angle_idx=i, contrast=0.3, offset=1.0)
            )
            for i, p in enumerate(phi)
        ]
    )
    c2[0, 1, 2] += 1.0
    w = np.ones_like(c2)
    w[0, 1, 2] = 0.0
    d = build_heterodyne_stratified_data(model, c2, phi, w)
    residual, x, _, p0, _ = build_joint_pointwise_residual(
        model=model,
        stratified_data=d,
        per_angle_mode="individual",
        init_scaling=np.array([0.3, 0.3, 1.0, 1.0]),
    )
    sel = (np.asarray(x) == [0, 1, 2]).all(axis=1)
    assert sel.sum() == 1
    assert np.asarray(residual(p0))[sel].item() == 0.0


def test_opthet_b4():
    model, c2, phi = make_synthetic_two_component(n_phi=2, n_t=12)
    cfg = NLSQConfig(per_angle_mode="individual", regularization_mode="adaptive")
    p = _build_joint_problem(model, c2, phi, cfg, weights=None)
    assert p.meta["regularization_active"]
    jac = np.asarray(jax.jacfwd(p.joint_residual_fn)(jnp.asarray(p.x0)))
    assert np.isfinite(jac).all(), np.argwhere(~np.isfinite(jac))[:8]


def test_opthet_b5(monkeypatch):
    model, c2, phi = make_synthetic_two_component(n_phi=2, n_t=12)
    model.param_manager.set_vary("D0_ref", False)
    cfg = NLSQConfig(
        per_angle_mode="individual",
        enable_cmaes=True,
        cmaes_warmstart_auto_skip=False,
    )
    backend = Mock(return_value=SimpleNamespace(success=False, parameters=None))
    monkeypatch.setattr(heterodyne_core, "fit_with_cmaes", backend)
    monkeypatch.setattr(heterodyne_core, "HAS_CMAES", True)
    heterodyne_core.fit_nlsq_multi_phi(model, c2, phi, cfg)
    assert backend.called
    assert backend.call_args.kwargs["p0"].size == 17


def test_opthet_b7():
    model, c2, phi = make_synthetic_two_component(n_phi=2, n_t=12)
    r = fit_two_component_via_engine(
        model,
        c2,
        phi,
        NLSQConfig(method="lm", loss="linear", per_angle_mode="individual"),
    )
    assert "doesn't work when there are bounds" not in r.nlsq_diagnostics["message"]


def test_opthet_b9():
    # Real solves (no stub): the two identically-zero averaged-mode penalty rows
    # leave the objective and trajectory unchanged, so the covariance must match.
    def run(mode):
        m, c2, phi = make_synthetic_two_component(n_phi=3, n_t=12)
        return fit_nlsq_multi_phi(m, c2, phi, NLSQConfig(regularization_mode=mode, loss="linear"))

    a, b = run("adaptive"), run("none")
    assert a.nlsq_diagnostics["per_angle_mode"] == "averaged"
    assert a.nlsq_diagnostics.get("regularization_active") is True
    np.testing.assert_allclose(a.chi_squared, b.chi_squared, rtol=1e-9)
    assert np.isfinite(np.diag(b.covariance)).all()
    # Today: ratio == 314/316 (nlsq divides by residual rows - p, rows include the 2 zeros).
    # Median over the diagonal: a few ill-conditioned entries are noisy on their own.
    ratio = float(np.median(np.diag(a.covariance) / np.diag(b.covariance)))
    assert abs(ratio - 1.0) < 1e-3, (ratio, 314 / 316)


# opthet-B10 reclassified DESIGN by the lead (with optstrat-B9): regularization.enable is ignored by
# design in both models (pinned by test_l3_banner_always_on_even_when_regularization_disabled); the
# shipped template has enable: false + mode: relative, so honouring it changes default behaviour.


def test_opthet_b12():
    model, _c2, phi = make_synthetic_two_component(n_phi=3, n_t=12)
    n = model.param_manager.n_varying
    # Real backend text: what NLSQAdapter.fit actually returns on budget exhaustion.
    real = NLSQAdapter(["x", "y"]).fit(
        residual_fn=lambda p: jnp.stack([10 * (p[1] - p[0] ** 2), 1 - p[0], p[0] * p[1] - 1.0]),
        initial_params=np.array([-1.2, 1.0]),
        bounds=(np.array([-9.0, -9.0]), np.array([9.0, 9.0])),
        config=NLSQConfig(loss="linear", max_nfev=2),
    )
    assert not real.success
    res = build_hybrid_streaming_result(
        model=model,
        popt=np.zeros(n),
        pcov=np.eye(n),
        info={
            "nit": 1513,
            "success": False,
            "convergence_reason": real.convergence_reason,
            "cost": 5.0,
            "n_data_points": 1000,
            "sigma2_noise": 0.01,
        },
        phi_angles=phi,
    )
    assert res.reduced_chi_squared < 2.0
    assert res.convergence_status == "max_iter", real.convergence_reason
    assert res.quality_flag == "good"
