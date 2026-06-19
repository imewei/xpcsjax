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
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "individual"})
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

    hc._fit_joint_cmaes_multi_phi(model=model, c2_data=c2, phi_angles=phi, config=cfg, weights=None)

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
# Success flag on floor-revert (parity BLOCKER, 2026-06-16). When the floor
# rejects a degraded solve and reverts to x0 it drops ``joint_result`` (None) to
# discard the rejected vector's covariance. ``_build_joint_result`` previously
# read ``solve_success = ... else True`` on that None, so a reverted (i.e.
# NON-converged) warm-start reported success=True. The joint CMA-ES auto-skip
# reads ``warm.success`` (heterodyne_core.py:1721) — a spurious True there
# auto-skips the very CMA-ES escape the success-gate exists to run on the
# degenerate C044 ``two_component`` Jacobian, defeating laminar parity
# (core.py:2320 sets warm params/chi2 ONLY on the solver's success flag).
# An assembly NOT backed by a global escape (revert path) must report failure.
# ===========================================================================
def test_floor_revert_reports_failure_not_spurious_success(monkeypatch):
    """A reverted-to-x0 warm-start must report success=False / 'failed'.

    Exercises the REAL ``_build_joint_result`` (not a synthetic ``warm_success``
    flag) through the production revert path: degraded solve → floor revert →
    ``joint_result=None`` → result assembly. Pre-fix this reported success=True,
    silently feeding the CMA-ES auto-skip gate a converged verdict on a
    non-converged warm-start.
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

    res = hc._fit_joint_multi_phi(model, c2, phi, cfg, None)

    assert res.success is False, (
        "a warm-start that reverted to x0 must NOT report success=True — a "
        "spurious True defeats the CMA-ES auto-skip success-gate (parity with "
        "laminar core.py:2320)"
    )
    assert res.convergence_status == "failed"


def test_escape_kept_assembly_still_reports_success():
    """Blast-radius boundary: an assembly backed by a global escape (joint_result
    is None but ``global_escape`` is set) still reports success=True — the escape
    pre-accepted the vector. Only the no-escape revert path flips to False, so the
    documented escape-result contract (kept escape carries a 'success' verdict) is
    preserved. Guards the fix from over-reaching into the escape paths.
    """
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=16)
    cfg = _individual_cfg()
    prob = hc._build_joint_problem(model, c2, phi, cfg, weights=None)

    res = hc._build_joint_result(
        model,
        prob,
        c2,
        np.asarray(prob.x0, dtype=np.float64),
        phi,
        cfg,
        None,
        joint_result=None,
        global_escape="cmaes_warmstart_kept",
    )

    assert res.success is True
    assert res.convergence_status == "converged"


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
        assert not any(isinstance(f, hc._WarmStartProbeNoiseFilter) for f in lg.filters), (
            f"{name} still carries the probe noise filter after exit"
        )


# ===========================================================================
# Keep-better floor FALLBACK preserves Stage-1's per-angle scaling (C044
# two_component RCA, 2026-06-17). ``_build_joint_problem`` seeds the individual
# joint ``x0`` scaling head by broadcasting the SCALAR ``model.scaling.contrast[0]``
# across all angles (legacy mirror). When Stage-1's quantile fit froze DISTINCT
# per-angle scaling (real multi-angle data), that broadcast discards it, so the
# keep-better floor's only fallback (``x0``) is as degraded as the failed solve —
# and a degenerate joint fit finalized the WORSE Stage-2 vector (C044: 25663 vs
# Stage-1 6796). The floor now reverts to the BEST of {x0, a fallback that
# preserves the per-angle scaling = Stage-1's actual fit}. Strictly keep-better,
# so the SUCCESS path and the uniform-scaling fixtures above are untouched.
# ===========================================================================
def test_build_floor_fallback_preserves_distinct_per_angle_scaling():
    """The fallback reconstructs ``[per-angle scaling | physics]`` (NOT the
    scalar-broadcast x0) when the frozen scaling is distinct across angles."""
    n_phi, n_physics = 3, 4
    per_angle_contrast = np.array([0.20, 0.35, 0.50], dtype=np.float64)
    per_angle_offset = np.array([1.00, 1.05, 0.95], dtype=np.float64)
    physics_initial = np.array([10.0, 0.5, -0.3, 100.0], dtype=np.float64)
    # x0 collapses every angle to angle-0's scalar (the bug this fallback fixes).
    x0 = np.concatenate([np.full(n_phi, 0.20), np.full(n_phi, 1.00), physics_initial])
    lb = np.concatenate([np.full(2 * n_phi, -10.0), np.full(n_physics, -1e6)])
    ub = np.concatenate([np.full(2 * n_phi, 10.0), np.full(n_physics, 1e6)])

    fb = hc._build_floor_fallback_x0(
        resolved_mode="individual",
        n_phi=n_phi,
        n_physics_varying=n_physics,
        per_angle_contrast=per_angle_contrast,
        per_angle_offset=per_angle_offset,
        physics_initial=physics_initial,
        lb=lb,
        ub=ub,
        x0=x0,
    )

    np.testing.assert_allclose(fb[:n_phi], per_angle_contrast)
    np.testing.assert_allclose(fb[n_phi : 2 * n_phi], per_angle_offset)
    np.testing.assert_allclose(fb[2 * n_phi :], physics_initial)
    assert not np.allclose(fb, x0), "fallback must differ from the collapsed x0"


def test_build_floor_fallback_is_x0_for_uniform_scaling():
    """Uniform per-angle scaling → fallback IS x0 (the broadcast loses nothing)."""
    n_phi, n_physics = 3, 2
    uniform_c = np.full(n_phi, 0.30, dtype=np.float64)
    uniform_o = np.full(n_phi, 1.00, dtype=np.float64)
    physics_initial = np.array([5.0, -1.0], dtype=np.float64)
    x0 = np.concatenate([uniform_c, uniform_o, physics_initial])
    lb = np.concatenate([np.full(2 * n_phi, -10.0), np.full(n_physics, -1e6)])
    ub = np.concatenate([np.full(2 * n_phi, 10.0), np.full(n_physics, 1e6)])

    fb = hc._build_floor_fallback_x0(
        resolved_mode="individual",
        n_phi=n_phi,
        n_physics_varying=n_physics,
        per_angle_contrast=uniform_c,
        per_angle_offset=uniform_o,
        physics_initial=physics_initial,
        lb=lb,
        ub=ub,
        x0=x0,
    )
    np.testing.assert_array_equal(fb, x0)


def test_build_floor_fallback_is_x0_for_non_individual_modes():
    """``averaged`` / ``constant`` have no scalar-broadcast collapse → pass x0 through."""
    x0 = np.array([0.3, 1.0, 5.0, -1.0], dtype=np.float64)
    lb = np.full_like(x0, -1e6)
    ub = np.full_like(x0, 1e6)
    for mode in ("averaged", "constant"):
        fb = hc._build_floor_fallback_x0(
            resolved_mode=mode,
            n_phi=3,
            n_physics_varying=2,
            per_angle_contrast=np.array([0.2, 0.35, 0.5]),
            per_angle_offset=np.array([1.0, 1.05, 0.95]),
            physics_initial=np.array([5.0, -1.0]),
            lb=lb,
            ub=ub,
            x0=x0,
        )
        np.testing.assert_array_equal(fb, x0, err_msg=f"mode={mode} must pass x0 through")


def test_build_joint_problem_exposes_floor_fallback():
    """``_build_joint_problem`` must publish a ``floor_fallback_x0`` of x0's shape."""
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=16)
    prob = hc._build_joint_problem(model, c2, phi, _individual_cfg(), weights=None)
    assert "floor_fallback_x0" in prob.meta, "joint problem must expose floor_fallback_x0"
    fb = np.asarray(prob.meta["floor_fallback_x0"], dtype=np.float64)
    assert fb.shape == np.asarray(prob.x0, dtype=np.float64).shape


def test_floor_reverts_to_better_fallback_not_degraded_x0(monkeypatch):
    """When x0 is degraded and the per-angle fallback is strictly better, the
    floor reverts to the FALLBACK (Stage-1's fit), not the degraded x0.

    Deterministic: x0 is overridden to a deliberately-suboptimal point and the
    fallback is set to the near-optimal warm-start, so the keep-better revert has
    an unambiguous winner without depending on a real degenerate C044 solve.
    """
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=16)
    cfg = _individual_cfg()

    prob = hc._build_joint_problem(model, c2, phi, cfg, weights=None)
    good = np.asarray(prob.x0, dtype=np.float64)  # near-optimal warm-start
    base = prob.meta["base_residual_fn"]

    def _ssr(x: np.ndarray) -> float:
        return float(np.sum(np.asarray(base(x), dtype=np.float64) ** 2))

    bad = np.clip(good + 0.30 * (prob.ub - prob.lb), prob.lb, prob.ub)  # worse x0
    assert _ssr(bad) > _ssr(good), "precondition: overridden x0 must be worse than fallback"
    prob.meta["floor_fallback_x0"] = good  # fallback preserves the good point

    degraded = np.clip(good + 0.60 * (prob.ub - prob.lb), prob.lb, prob.ub)
    assert _ssr(degraded) > _ssr(bad), "precondition: solve degrades vs x0"
    monkeypatch.setattr(hc, "NLSQAdapter", _degraded_adapter(degraded, _ssr(degraded)))

    captured: dict[str, np.ndarray] = {}

    def _capture_build_result(*args, **_kwargs):
        captured["params"] = np.asarray(args[3], dtype=np.float64)
        return "SENTINEL"

    monkeypatch.setattr(hc, "_build_joint_result", _capture_build_result)

    hc._fit_joint_multi_phi(model, c2, phi, cfg, None, x0_override=bad, prob=prob)

    np.testing.assert_allclose(
        captured["params"],
        good,
        err_msg="degraded solve must floor to the better per-angle fallback, not x0",
    )


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
        with caplog.at_level(logging.DEBUG, logger="xpcsjax.optimization.nlsq.heterodyne_core"):
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
