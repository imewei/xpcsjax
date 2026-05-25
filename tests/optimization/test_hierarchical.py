"""Tests for xpcsjax.optimization.nlsq.hierarchical.

The two-stage optimizer alternates L-BFGS-B over frozen per-angle / physical
splits. The parameter-freezing closures are tested directly with deterministic
loss/grad functions (exact buffer assembly), and the full ``fit`` is driven
end-to-end on a separable quadratic where each stage has a known minimizer, so
convergence and loss reduction are verifiable without real XPCS physics.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from xpcsjax.optimization.nlsq import hierarchical as h

# ---------------------------------------------------------------------------
# HierarchicalConfig / HierarchicalResult
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = h.HierarchicalConfig()
    assert cfg.enable is True
    assert cfg.max_outer_iterations == 5
    assert cfg.outer_tolerance == 1e-6


def test_config_from_dict() -> None:
    cfg = h.HierarchicalConfig.from_dict(
        {
            "enable": False,
            "max_outer_iterations": 3,
            "outer_tolerance": 1e-4,
            "physical_max_iterations": 20,
            "per_angle_ftol": 1e-5,
            "log_stage_transitions": False,
        }
    )
    assert cfg.enable is False
    assert cfg.max_outer_iterations == 3
    assert cfg.outer_tolerance == 1e-4
    assert cfg.physical_max_iterations == 20
    assert cfg.per_angle_ftol == 1e-5
    assert cfg.log_stage_transitions is False


def test_result_dataclass() -> None:
    res = h.HierarchicalResult(x=np.zeros(3), fun=1.0, success=True, n_outer_iterations=2)
    assert res.history == []
    assert res.total_time == 0.0


# ---------------------------------------------------------------------------
# __init__ index layout
# ---------------------------------------------------------------------------


def test_optimizer_indices_non_fourier() -> None:
    opt = h.HierarchicalOptimizer(h.HierarchicalConfig(), n_phi=3, n_physical=2)
    assert opt.n_per_angle == 6  # 2 * n_phi
    np.testing.assert_array_equal(opt.per_angle_indices, [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(opt.physical_indices, [6, 7])


def test_optimizer_indices_fourier() -> None:
    fourier = cast(Any, SimpleNamespace(n_coeffs=10))
    opt = h.HierarchicalOptimizer(
        h.HierarchicalConfig(), n_phi=23, n_physical=7, fourier_reparameterizer=fourier
    )
    assert opt.n_per_angle == 10  # n_coeffs, not 2 * n_phi
    np.testing.assert_array_equal(opt.physical_indices, np.arange(10, 17))


# ---------------------------------------------------------------------------
# parameter-freezing closures (deterministic buffer assembly)
# ---------------------------------------------------------------------------


def _opt() -> h.HierarchicalOptimizer:
    # n_phi=1 -> per_angle indices [0, 1], physical indices [2, 3].
    return h.HierarchicalOptimizer(h.HierarchicalConfig(), n_phi=1, n_physical=2)


def test_create_physical_loss_assembles_full_vector() -> None:
    opt = _opt()

    def loss(full: np.ndarray) -> float:
        return float(full[0] * 1 + full[1] * 10 + full[2] * 100 + full[3] * 1000)

    phys_loss = opt._create_physical_loss(np.array([1.0, 2.0]), loss)
    # buffer = [1, 2, 3, 4] -> 1 + 20 + 300 + 4000
    assert phys_loss(np.array([3.0, 4.0])) == 4321.0


def test_create_physical_grad_slices_physical_indices() -> None:
    opt = _opt()

    def grad(_full: np.ndarray) -> np.ndarray:
        return np.arange(4.0)  # [0, 1, 2, 3]

    pg = opt._create_physical_grad(np.zeros(2), grad)
    np.testing.assert_array_equal(pg(np.zeros(2)), [2.0, 3.0])  # physical idx [2, 3]


def test_create_per_angle_loss_and_grad() -> None:
    opt = _opt()

    def loss(full: np.ndarray) -> float:
        return float(np.sum(full))

    pa_loss = opt._create_per_angle_loss(np.array([9.0, 9.0]), loss)
    # frozen physical = [9, 9]; per-angle = [1, 1] -> sum = 20
    assert pa_loss(np.array([1.0, 1.0])) == 20.0

    def grad(_full: np.ndarray) -> np.ndarray:
        return np.arange(4.0)

    pg = opt._create_per_angle_grad(np.zeros(2), grad)
    np.testing.assert_array_equal(pg(np.zeros(2)), [0.0, 1.0])  # per-angle idx [0, 1]


# ---------------------------------------------------------------------------
# get_diagnostics
# ---------------------------------------------------------------------------


def test_get_diagnostics() -> None:
    opt = h.HierarchicalOptimizer(h.HierarchicalConfig(), n_phi=2, n_physical=3)
    diag = opt.get_diagnostics()
    assert diag["n_phi"] == 2
    assert diag["n_physical"] == 3
    assert diag["n_per_angle"] == 4
    assert diag["fourier_enabled"] is False


# ---------------------------------------------------------------------------
# fit() end-to-end on a separable quadratic
# ---------------------------------------------------------------------------


def test_fit_converges_on_separable_quadratic() -> None:
    # 4 params: per-angle [0,1], physical [2,3]. Each stage minimizes toward
    # its slice of `target`, so the two-stage alternation drives loss -> 0.
    target = np.array([0.5, 0.6, 0.7, 0.8])

    def loss(p: np.ndarray) -> float:
        return float(np.sum((np.asarray(p) - target) ** 2))

    def grad(p: np.ndarray) -> np.ndarray:
        return 2.0 * (np.asarray(p) - target)

    opt = h.HierarchicalOptimizer(
        h.HierarchicalConfig(max_outer_iterations=5, log_stage_transitions=False),
        n_phi=1,
        n_physical=2,
    )
    p0 = np.zeros(4)
    bounds = (np.full(4, -10.0), np.full(4, 10.0))

    result = opt.fit(loss, grad, p0, bounds)

    assert isinstance(result.x, np.ndarray)
    assert result.fun < loss(p0)  # loss reduced
    assert result.fun < 1e-3  # converged near the separable minimum
    np.testing.assert_allclose(result.x, target, atol=1e-2)
    assert result.n_outer_iterations >= 1
    assert len(result.history) == result.n_outer_iterations


def test_fit_invokes_outer_callback_and_logs() -> None:
    target = np.array([0.1, 0.2, 0.3, 0.4])

    def loss(p: np.ndarray) -> float:
        return float(np.sum((np.asarray(p) - target) ** 2))

    def grad(p: np.ndarray) -> np.ndarray:
        return 2.0 * (np.asarray(p) - target)

    # log_stage_transitions=True exercises the per-stage logging branches.
    opt = h.HierarchicalOptimizer(
        h.HierarchicalConfig(max_outer_iterations=2, log_stage_transitions=True),
        n_phi=1,
        n_physical=2,
    )
    seen: list[int] = []
    opt.fit(
        loss,
        grad,
        np.zeros(4),
        (np.full(4, -5.0), np.full(4, 5.0)),
        outer_iteration_callback=lambda _p, i: seen.append(i),
    )
    assert seen[0] == 0  # callback fired starting at outer iteration 0
