"""Regression tests for the 2026-07-22 debug-audit fixes.

Fix 1 (heterodyne_core.py / heterodyne_constant_mode.py): the L2 keep-better
SSR floor that ``_fit_joint_multi_phi`` (individual mode) already had was
extracted into ``_apply_joint_keep_better_floor`` and wired into the sibling
averaged/constant joint solves, which previously had NO such protection.

Fix 2 (heterodyne_core.py ``_build_joint_problem``, UNCERTAIN — flagged by the
audit verifier, apply with extra scrutiny): hardcoded per-angle contrast
bounds ``(0.1, 0.8)`` now source from ``SCALING_PARAMS`` like every sibling
per-angle-mode solver.

Fix 3 (heterodyne_stratified_ls.py): ``scaling_names`` for individual mode is
now built from the deduplicated angle count (matching the actual scaling
vector), not the raw ``len(phi)``.

Fix 4 (recovery.py): ``diagnose_error``'s convergence_failure/unknown_error
perturbation branches now use the same absolute-value-floor perturb scale as
the stagnation-retry path, so a zero-valued parameter can still be perturbed.
"""

from __future__ import annotations

import numpy as np

from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.config.parameter_registry import SCALING_PARAMS
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_constant_mode import (
    _fit_joint_constant_multi_phi,
)
from xpcsjax.optimization.nlsq.heterodyne_core import (
    _apply_joint_keep_better_floor,
    _build_joint_problem,
    _fit_joint_averaged_multi_phi,
    _fit_joint_multi_phi,
)

# ---------------------------------------------------------------------------
# Fix 1: shared keep-better SSR floor
# ---------------------------------------------------------------------------


def _linear_residual_fn(target: np.ndarray):
    """Residual = x - target; SSR is minimized (0) exactly at x == target."""

    def _fn(x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) - target

    return _fn


def test_keep_better_floor_keeps_a_better_candidate():
    x0 = np.array([1.0, 1.0])
    target = np.array([0.0, 0.0])  # candidate below is closer to target -> lower SSR
    resid = _linear_residual_fn(target)
    candidate = np.array([0.1, 0.1])
    final, reverted = _apply_joint_keep_better_floor(resid, x0, candidate)
    np.testing.assert_allclose(final, candidate)
    assert reverted is False


def test_keep_better_floor_reverts_to_x0_when_candidate_is_worse():
    x0 = np.array([0.1, 0.1])
    target = np.array([0.0, 0.0])
    resid = _linear_residual_fn(target)
    degraded_candidate = np.array([5.0, 5.0])  # far worse than x0
    final, reverted = _apply_joint_keep_better_floor(resid, x0, degraded_candidate)
    np.testing.assert_allclose(final, x0)
    assert reverted is True


def test_keep_better_floor_prefers_a_better_fallback_over_x0():
    x0 = np.array([5.0, 5.0])
    target = np.array([0.0, 0.0])
    resid = _linear_residual_fn(target)
    degraded_candidate = np.array([10.0, 10.0])
    good_fallback = np.array([0.2, 0.2])
    final, reverted = _apply_joint_keep_better_floor(
        resid, x0, degraded_candidate, floor_fallback_x0=good_fallback
    )
    np.testing.assert_allclose(final, good_fallback)
    assert reverted is True


def test_keep_better_floor_never_regresses_ssr():
    """No-worse contract: final SSR is always <= the warm-start SSR."""
    target = np.array([0.0, 0.0, 0.0])
    resid = _linear_residual_fn(target)
    x0 = np.array([1.0, 1.0, 1.0])
    ssr_x0 = float(np.sum(resid(x0) ** 2))
    rng = np.random.default_rng(0)
    for _ in range(20):
        candidate = x0 + rng.uniform(-5, 5, size=3)
        final, _ = _apply_joint_keep_better_floor(resid, x0, candidate)
        ssr_final = float(np.sum(resid(final) ** 2))
        assert ssr_final <= ssr_x0 * (1.0 + 1e-9)


def test_averaged_mode_now_wires_the_keep_better_floor(monkeypatch):
    """Previously ``_fit_joint_averaged_multi_phi`` never called the floor —
    this pins that the propagation actually happened."""
    calls = []
    real = _apply_joint_keep_better_floor

    def _spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(
        "xpcsjax.optimization.nlsq.heterodyne_core._apply_joint_keep_better_floor", _spy
    )
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=16)
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "averaged"})
    result = _fit_joint_averaged_multi_phi(model, c2, phi, cfg, weights=None)
    assert calls, "averaged-mode joint solve must call the keep-better floor"
    assert np.isfinite(result.chi_squared)


def test_constant_mode_now_wires_the_keep_better_floor(monkeypatch):
    """Previously ``_fit_joint_constant_multi_phi`` never called the floor —
    this pins that the propagation actually happened."""
    calls = []
    real = _apply_joint_keep_better_floor

    def _spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    # heterodyne_constant_mode imports the helper lazily from heterodyne_core
    # at call time, so patching the heterodyne_core attribute is what matters.
    monkeypatch.setattr(
        "xpcsjax.optimization.nlsq.heterodyne_core._apply_joint_keep_better_floor", _spy
    )
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=16)
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "constant"})
    result = _fit_joint_constant_multi_phi(model, c2, phi, cfg, weights=None)
    assert calls, "constant-mode joint solve must call the keep-better floor"
    assert np.isfinite(result.chi_squared)


def test_individual_mode_still_wires_the_floor_unchanged(monkeypatch):
    """No-regression: individual mode already called the floor pre-extraction;
    confirm the refactor kept it wired."""
    calls = []
    real = _apply_joint_keep_better_floor

    def _spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(
        "xpcsjax.optimization.nlsq.heterodyne_core._apply_joint_keep_better_floor", _spy
    )
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=16)
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "individual"})
    result = _fit_joint_multi_phi(model, c2, phi, cfg, weights=None)
    assert calls
    assert np.isfinite(result.chi_squared)


# ---------------------------------------------------------------------------
# Fix 2 (UNCERTAIN — flagged by the audit verifier, extra scrutiny requested):
# ``_build_joint_problem``'s scaling-first bounds must source from the
# parameter registry, not a hardcoded (0.1, 0.8) contrast literal.
# ---------------------------------------------------------------------------


def test_build_joint_problem_individual_bounds_match_registry():
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=16)
    n_phi = len(phi)
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "individual"})
    prob = _build_joint_problem(model, c2, phi, cfg, weights=None)
    lb = np.asarray(prob.lb, dtype=np.float64)
    ub = np.asarray(prob.ub, dtype=np.float64)
    # Scaling-first layout: [contrast_0..N-1 | offset_0..N-1 | physics].
    contrast_lb, contrast_ub = lb[:n_phi], ub[:n_phi]
    offset_lb, offset_ub = lb[n_phi : 2 * n_phi], ub[n_phi : 2 * n_phi]

    contrast_info = SCALING_PARAMS["contrast"]
    offset_info = SCALING_PARAMS["offset"]
    np.testing.assert_allclose(contrast_lb, contrast_info.min_bound)
    np.testing.assert_allclose(contrast_ub, contrast_info.max_bound)
    np.testing.assert_allclose(offset_lb, offset_info.min_bound)
    np.testing.assert_allclose(offset_ub, offset_info.max_bound)
    # Pin the specific regression: the old hardcoded contrast upper bound
    # (0.8) was tighter than the registry's 1.0 and must no longer appear.
    assert not np.allclose(contrast_ub, 0.8)


# ---------------------------------------------------------------------------
# Fix 3: stratified-LS individual-mode scaling_names must match the
# deduplicated angle count actually used by the scaling vector.
# ---------------------------------------------------------------------------


def test_stratified_ls_scaling_names_match_dedup_phi_count():
    from xpcsjax.optimization.nlsq.heterodyne_core import _resolve_effective_mode
    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
        fit_heterodyne_stratified_least_squares,
    )

    model, c2_u, phi_u = make_synthetic_two_component(n_phi=3, n_t=16)
    # Introduce a genuine duplicate angle: n_phi (raw) = 4, unique = 3.
    dup = 0
    phi = np.concatenate([phi_u, phi_u[dup : dup + 1]])
    c2 = np.concatenate([c2_u, c2_u[dup : dup + 1]], axis=0)
    assert len(phi) != len(np.unique(phi))

    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "individual"})
    assert _resolve_effective_mode(cfg, len(phi)) == "individual"

    result = fit_heterodyne_stratified_least_squares(
        model=model, c2=c2, phi=phi, config=cfg, weights=None, shuffle=False
    )

    n_phi_unique = len(np.unique(phi))
    n_physics = int(model.param_manager.n_varying)
    expected_n_params = n_physics + 2 * n_phi_unique
    assert len(result.parameters) == expected_n_params

    param_names = result.nlsq_diagnostics["parameter_names"]
    assert len(param_names) == len(result.parameters), (
        "scaling_names (feeding joint_param_names) must match the actual "
        f"scaling vector length: got {len(param_names)} names for "
        f"{len(result.parameters)} parameters"
    )


# ---------------------------------------------------------------------------
# Fix 4: recovery.py's convergence_failure / unknown_error perturbation must
# be able to move a zero-valued parameter (additive floor, not multiplicative).
# ---------------------------------------------------------------------------


def test_diagnose_error_convergence_failure_perturbs_zero_param():
    from xpcsjax.optimization.nlsq.recovery import diagnose_error

    params = np.array([0.0, 3.0, -2.0])
    diag = diagnose_error(
        error=ValueError("convergence failure: max iterations reached"),
        params=params,
        bounds=None,
        attempt=0,
    )
    new_params = diag["recovery_strategy"]["new_params"]
    assert new_params[0] != 0.0, "zero-valued parameter must receive a non-zero perturbation"


def test_diagnose_error_convergence_failure_retry_perturbs_zero_param():
    from xpcsjax.optimization.nlsq.recovery import diagnose_error

    params = np.array([0.0, 3.0, -2.0])
    diag = diagnose_error(
        error=ValueError("convergence failure: max iterations reached"),
        params=params,
        bounds=None,
        attempt=1,
    )
    new_params = diag["recovery_strategy"]["new_params"]
    assert new_params[0] != 0.0


def test_diagnose_error_unknown_error_perturbs_zero_param():
    from xpcsjax.optimization.nlsq.recovery import diagnose_error

    params = np.array([0.0, 3.0, -2.0])
    diag = diagnose_error(
        error=ValueError("some completely unrecognized failure"),
        params=params,
        bounds=None,
        attempt=0,
    )
    new_params = diag["recovery_strategy"]["new_params"]
    assert new_params[0] != 0.0
