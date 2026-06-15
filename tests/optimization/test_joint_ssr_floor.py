"""Keep-better floor on the heterodyne joint solve + Stage-1 de-duplication.

Pins two robustness fixes surfaced by the C044 ``two_component`` RCA
(``xpcsjax_two_component_20260615_015121.log``):

1. **SSR floor.** A degenerate joint solve (a ``nan``-gradient trust-region step
   on the near-singular C044 Jacobian) returned parameters with a HIGHER
   data-only SSR than its own warm-start ``x0`` (6796 → 25663). That degraded
   vector then became the CMA-ES warm-start, inflating the final result. A
   trust-region solve must never return worse than its start; ``_fit_joint_multi_phi``
   now reverts to ``x0`` when the solve degrades the data-only SSR. SSR-monotone,
   so parity-safe under the ``two_component`` no-worse contract.

2. **Stage-1 de-duplication.** ``_build_joint_problem`` runs the expensive
   Stage-1 constant-mode solve; the CMA-ES escape called it twice (once directly,
   once again inside ``_fit_joint_multi_phi``). The escape now threads its
   already-built ``prob`` down so Stage 1 runs once.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np

from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq import heterodyne_core as hc
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig


def _individual_cfg() -> NLSQConfig:
    cfg = NLSQConfig.from_dict(
        {"analysis_mode": "two_component", "per_angle_mode": "individual"}
    )
    # Keep the L4 jax-grad monitor out of the unit test (observational only).
    cfg.enable_gradient_monitoring = False
    return cfg


def test_joint_fit_reverts_to_x0_when_backend_degrades_ssr(monkeypatch):
    """A solve that increases data-only SSR vs x0 must be floored back to x0."""
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=16)
    cfg = _individual_cfg()

    prob = hc._build_joint_problem(model, c2, phi, cfg, weights=None)
    x0 = np.asarray(prob.x0, dtype=np.float64)
    base = prob.meta["base_residual_fn"]

    def _ssr(x: np.ndarray) -> float:
        return float(np.sum(np.asarray(base(x), dtype=np.float64) ** 2))

    # A deliberately degraded (but in-bounds) vector with SSR strictly worse
    # than the warm-start — the failure mode the floor must catch.
    degraded = np.clip(x0 + 0.5 * (prob.ub - prob.lb), prob.lb, prob.ub)
    assert _ssr(degraded) > _ssr(x0), "test precondition: degraded must be worse"

    class _FakeAdapter:
        def __init__(self, parameter_names):
            self.parameter_names = parameter_names

        def fit(self, **_kwargs):
            # Reports success yet returns a worse-than-x0 vector: proves the
            # floor is driven by SSR, not by the solver's success flag.
            return SimpleNamespace(
                success=True,
                message="ok",
                parameters=degraded,
                uncertainties=None,
                covariance=None,
                chi_squared=_ssr(degraded),
                n_iterations=1,
            )

    monkeypatch.setattr(hc, "NLSQAdapter", _FakeAdapter)

    captured: dict[str, np.ndarray] = {}

    def _capture_build_result(*args, **_kwargs):
        captured["params"] = np.asarray(args[3], dtype=np.float64)
        return "SENTINEL"

    monkeypatch.setattr(hc, "_build_joint_result", _capture_build_result)

    out = hc._fit_joint_multi_phi(model, c2, phi, cfg, None)

    assert out == "SENTINEL"
    np.testing.assert_allclose(
        captured["params"], x0, err_msg="degraded solve must be floored to x0"
    )


def test_joint_fit_keeps_non_degrading_solve(monkeypatch):
    """The floor is a no-op when the solve does not increase SSR (tie at x0).

    Guards against an over-eager floor that mutates a healthy solve. On the
    noise-free synthetic fixture x0 already sits at the optimum, so the
    boundary case (solve returns a vector whose SSR equals x0) is the
    achievable no-op to pin: the returned vector must pass through untouched.
    """
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=16)
    cfg = _individual_cfg()

    prob = hc._build_joint_problem(model, c2, phi, cfg, weights=None)
    x0 = np.asarray(prob.x0, dtype=np.float64)
    base = prob.meta["base_residual_fn"]
    ssr_x0 = float(np.sum(np.asarray(base(x0), dtype=np.float64) ** 2))

    class _FakeAdapter:
        def __init__(self, parameter_names):
            self.parameter_names = parameter_names

        def fit(self, **_kwargs):
            return SimpleNamespace(
                success=True,
                message="ok",
                parameters=x0.copy(),
                uncertainties=None,
                covariance=None,
                chi_squared=ssr_x0,
                n_iterations=3,
            )

    monkeypatch.setattr(hc, "NLSQAdapter", _FakeAdapter)

    captured: dict[str, np.ndarray] = {}

    def _capture_build_result(*args, **_kwargs):
        captured["params"] = np.asarray(args[3], dtype=np.float64)
        return "SENTINEL"

    monkeypatch.setattr(hc, "_build_joint_result", _capture_build_result)

    hc._fit_joint_multi_phi(model, c2, phi, cfg, None)

    np.testing.assert_allclose(
        captured["params"], x0, err_msg="non-degrading solve must pass through"
    )


def test_cmaes_escape_builds_joint_problem_once(monkeypatch):
    """The CMA-ES escape builds the joint problem (incl. Stage 1) exactly once.

    Before the fix the escape built ``prob`` directly AND ``_fit_joint_multi_phi``
    rebuilt it internally — running the expensive Stage-1 constant-mode solve
    twice. The fix threads the escape's ``prob`` into the warm-start joint fit so
    ``_build_joint_problem`` runs once. CMA-ES is stubbed to a no-win so the real
    global optimizer never runs; the real warm-start joint fit still executes.
    """
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=16)
    cfg = _individual_cfg()

    real_build = hc._build_joint_problem
    n_builds = {"count": 0}

    def _counting_build(*args, **kwargs):
        n_builds["count"] += 1
        return real_build(*args, **kwargs)

    monkeypatch.setattr(hc, "_build_joint_problem", _counting_build)
    # CMA-ES never wins → escape keeps the warm-start; keeps the real optimizer
    # (and its minutes-long global search) out of the unit test.
    monkeypatch.setattr(
        hc, "fit_with_cmaes", lambda **_kwargs: SimpleNamespace(success=False, parameters=None)
    )

    hc._fit_joint_cmaes_multi_phi(
        model=model, c2_data=c2, phi_angles=phi, config=cfg, weights=None
    )

    assert n_builds["count"] == 1, (
        f"Stage-1/joint problem built {n_builds['count']}x; expected 1 "
        "(escape must reuse its prebuilt prob)"
    )


def test_floor_revert_discards_degraded_solve_covariance(monkeypatch):
    """On revert to x0 the result must NOT carry the rejected solve's covariance.

    When the floor rejects a degraded-but-finite-covariance solve and reverts to
    x0, the returned OptimizationResult describes x0 — so its covariance and
    uncertainties (which were computed at the discarded vector) must be dropped,
    not copied through. Exercises the REAL ``_build_joint_result`` (not stubbed).
    """
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=16)
    cfg = _individual_cfg()

    prob = hc._build_joint_problem(model, c2, phi, cfg, weights=None)
    x0 = np.asarray(prob.x0, dtype=np.float64)
    base = prob.meta["base_residual_fn"]

    def _ssr(x: np.ndarray) -> float:
        return float(np.sum(np.asarray(base(x), dtype=np.float64) ** 2))

    degraded = np.clip(x0 + 0.5 * (prob.ub - prob.lb), prob.lb, prob.ub)
    assert _ssr(degraded) > _ssr(x0), "test precondition: degraded must be worse"

    n = len(x0)
    finite_cov = np.eye(n, dtype=np.float64) * 0.123
    finite_unc = np.full(n, 0.456, dtype=np.float64)

    class _FakeAdapter:
        def __init__(self, parameter_names):
            self.parameter_names = parameter_names

        def fit(self, **_kwargs):
            # A solve that REPORTS success with FINITE covariance, yet degraded SSR.
            return SimpleNamespace(
                success=True,
                message="ok",
                parameters=degraded,
                uncertainties=finite_unc,
                covariance=finite_cov,
                chi_squared=_ssr(degraded),
                n_iterations=5,
            )

    monkeypatch.setattr(hc, "NLSQAdapter", _FakeAdapter)

    res = hc._fit_joint_multi_phi(model, c2, phi, cfg, None)

    cov = None if res.covariance is None else np.asarray(res.covariance, dtype=np.float64)
    unc = None if res.uncertainties is None else np.asarray(res.uncertainties, dtype=np.float64)
    assert cov is None or np.all(np.isnan(cov)), (
        "reverted-to-x0 result carried the rejected solve's covariance"
    )
    assert unc is None or np.all(np.isnan(unc)), (
        "reverted-to-x0 result carried the rejected solve's uncertainties"
    )


# ===========================================================================
# Honest warm-start-probe logging (fixed 2026-06-15). A CMA-ES / multistart
# warm-start probe that fails on a degenerate Jacobian is EXPECTED + recovered,
# so its sub-solver failure noise is demoted instead of screaming ERROR/WARNING.
# Pure logging/control-flow — numerics are unchanged.
# ===========================================================================
def test_quiet_warm_start_probe_logging_filters_only_known_noise(caplog):
    """The probe-quieting context drops ONLY known noise messages (not a blanket
    level mute): an unrelated record from the same logger still propagates, and
    the filter is fully removed on exit."""
    log = logging.getLogger("nlsq.curve_fit")  # in _WARM_START_PROBE_NOISE_LOGGERS
    noise = "Optimization failed to converge"  # matches a noise pattern
    unrelated = "an unexpected and unrelated diagnostic"  # must survive

    with caplog.at_level(logging.DEBUG, logger="nlsq.curve_fit"):
        with hc._quiet_warm_start_probe_logging():
            log.error(noise)
            log.error(unrelated)
        inside = [r.getMessage() for r in caplog.records]

    assert unrelated in inside, "unexpected records must NOT be suppressed by the probe filter"
    assert noise not in inside, "known probe noise must be dropped inside the context"

    # Filter removed on exit → the same noise message now propagates.
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="nlsq.curve_fit"):
        log.error(noise)
    assert any(noise in r.getMessage() for r in caplog.records), (
        "the noise filter must be removed when the probe context exits"
    )


def test_quiet_warm_start_probe_logging_filter_fully_detached():
    """No ``_WarmStartProbeNoiseFilter`` lingers on any probe logger after exit."""
    with hc._quiet_warm_start_probe_logging():
        pass
    for name in hc._WARM_START_PROBE_NOISE_LOGGERS:
        lg = logging.getLogger(name)
        assert not any(
            isinstance(f, hc._WarmStartProbeNoiseFilter) for f in lg.filters
        ), f"{name} still carries the probe noise filter after exit"


def _degraded_adapter(degraded, ssr):
    class _FakeAdapter:
        def __init__(self, parameter_names):
            self.parameter_names = parameter_names

        def fit(self, **_kwargs):
            return SimpleNamespace(
                success=True,
                message="ok",
                parameters=degraded,
                uncertainties=None,
                covariance=None,
                chi_squared=ssr,
                n_iterations=1,
            )

    return _FakeAdapter


def test_warm_start_probe_demotes_floor_revert_to_debug(monkeypatch, caplog):
    """The floor-revert message is WARNING for a standalone joint fit but DEBUG
    for a warm-start probe (where the revert is the designed, expected recovery).
    The numerics — the floored x0 — are identical either way.
    """
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=16)
    cfg = _individual_cfg()

    prob = hc._build_joint_problem(model, c2, phi, cfg, weights=None)
    x0 = np.asarray(prob.x0, dtype=np.float64)
    base = prob.meta["base_residual_fn"]

    def _ssr(x: np.ndarray) -> float:
        return float(np.sum(np.asarray(base(x), dtype=np.float64) ** 2))

    degraded = np.clip(x0 + 0.5 * (prob.ub - prob.lb), prob.lb, prob.ub)
    assert _ssr(degraded) > _ssr(x0), "test precondition: degraded must be worse"

    monkeypatch.setattr(hc, "NLSQAdapter", _degraded_adapter(degraded, _ssr(degraded)))

    msg_key = "degraded data-only SSR"

    def _revert_records(probe: bool):
        caplog.clear()
        with caplog.at_level(
            logging.DEBUG, logger="xpcsjax.optimization.nlsq.heterodyne_core"
        ):
            hc._fit_joint_multi_phi(model, c2, phi, cfg, None, warm_start_probe=probe)
        return [r for r in caplog.records if msg_key in r.getMessage()]

    standalone = _revert_records(False)
    probe = _revert_records(True)

    assert standalone and standalone[0].levelno == logging.WARNING, (
        "standalone joint fit must flag a degraded solve at WARNING"
    )
    assert probe and probe[0].levelno == logging.DEBUG, (
        "warm-start probe must demote the expected floor revert to DEBUG"
    )
