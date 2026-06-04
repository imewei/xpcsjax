"""Scientific/branch tests for xpcsjax.optimization.nlsq.multistart.

Pure functions (LHS/random sampling, custom-start merging, screening,
degeneracy clustering, worker counting) are validated by invariants:
samples land within bounds, identical seeds are reproducible, screening keeps
the lowest-cost starts, and degeneracy clustering separates distinct basins.
The orchestrator ``run_multistart_nlsq`` is driven through its in-process
sequential path (``n_workers=1``) with a trivial deterministic fit function,
covering ``_run_full_strategy`` and ``_OptimizeWorker`` without spawning.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from xpcsjax.optimization.nlsq import multistart as ms

# ---------------------------------------------------------------------------
# MultiStartConfig
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = ms.MultiStartConfig()
    assert cfg.enable is False
    assert cfg.n_starts == 10
    assert cfg.sampling_strategy == "latin_hypercube"


def test_config_from_nlsq_config() -> None:
    nlsq_cfg = SimpleNamespace(
        enable_multi_start=True,
        multi_start_n_starts=5,
        multi_start_seed=7,
        multi_start_sampling_strategy="random",
        multi_start_n_workers=2,
        multi_start_use_screening=False,
        multi_start_screen_keep_fraction=0.25,
        multi_start_refine_top_k=2,
        multi_start_refinement_ftol=1e-10,
        multi_start_degeneracy_threshold=0.2,
        multi_start_custom_starts=None,
    )
    cfg = ms.MultiStartConfig.from_nlsq_config(nlsq_cfg)
    assert cfg.enable is True
    assert cfg.n_starts == 5
    assert cfg.seed == 7
    assert cfg.sampling_strategy == "random"
    assert cfg.screen_keep_fraction == 0.25
    assert cfg.degeneracy_threshold == 0.2


def test_config_to_nlsq_global_config() -> None:
    pytest.importorskip("nlsq.global_optimization")
    cfg = ms.MultiStartConfig(
        n_starts=8,
        sampling_strategy="latin_hypercube",
        use_screening=True,
        screen_keep_fraction=0.5,
    )
    goc = cfg.to_nlsq_global_config()
    assert goc.n_starts == 8
    assert goc.sampler == "lhs"
    assert goc.elimination_rounds == 3  # screening on
    assert goc.elimination_fraction == pytest.approx(0.5)


def test_config_to_nlsq_global_config_screening_off() -> None:
    pytest.importorskip("nlsq.global_optimization")
    cfg = ms.MultiStartConfig(sampling_strategy="random", use_screening=False)
    goc = cfg.to_nlsq_global_config()
    assert goc.sampler == "lhs"  # random falls back to lhs
    assert goc.elimination_rounds == 0


# ---------------------------------------------------------------------------
# MultiStartResult.to_optimization_result
# ---------------------------------------------------------------------------


def _single(
    chi2: float, success: bool = True, cov: np.ndarray | None = None
) -> ms.SingleStartResult:
    return ms.SingleStartResult(
        start_idx=0,
        initial_params=np.array([1.0, 2.0]),
        final_params=np.array([1.0, 2.0]),
        chi_squared=chi2,
        reduced_chi_squared=chi2,
        success=success,
        covariance=cov,
    )


@pytest.mark.parametrize(
    ("rchi2", "expected_flag"),
    [(1.0, "good"), (5.0, "marginal"), (50.0, "poor")],
)
def test_to_optimization_result_quality_flags(rchi2: float, expected_flag: str) -> None:
    best = _single(rchi2, success=True, cov=np.eye(2))
    msr = ms.MultiStartResult(
        best=best, all_results=[best], config=ms.MultiStartConfig(), strategy_used="full"
    )
    opt = msr.to_optimization_result()
    assert opt.quality_flag == expected_flag
    assert opt.convergence_status == "converged"
    np.testing.assert_allclose(opt.uncertainties, np.ones(2))


def test_to_optimization_result_no_covariance_failed() -> None:
    best = _single(3.0, success=False, cov=None)
    msr = ms.MultiStartResult(
        best=best, all_results=[best], config=ms.MultiStartConfig(), strategy_used="full"
    )
    opt = msr.to_optimization_result()
    assert opt.convergence_status == "failed"
    np.testing.assert_array_equal(opt.covariance, np.eye(2))  # fallback identity
    np.testing.assert_array_equal(opt.uncertainties, np.zeros(2))  # fallback zeros


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------


def test_get_phi_from_data() -> None:
    assert ms._get_phi_from_data({"phi": [1.0, 2.0]}) is not None
    phi = ms._get_phi_from_data({"phi_angles_list": [0.0]})
    assert phi is not None
    assert np.array_equal(phi, np.array([0.0]))
    assert ms._get_phi_from_data({}) is None


def test_get_dataset_size() -> None:
    assert ms._get_dataset_size({"g2": np.zeros((2, 3, 3))}) == 18
    assert ms._get_dataset_size({"c2_exp": np.zeros((4, 4))}) == 16
    assert ms._get_dataset_size({"phi": np.zeros(7)}) == 7
    with pytest.raises(ValueError, match="Cannot determine dataset size"):
        ms._get_dataset_size({})


# ---------------------------------------------------------------------------
# bounds + n_starts validation
# ---------------------------------------------------------------------------


def test_check_zero_volume_bounds() -> None:
    assert ms.check_zero_volume_bounds(np.array([[1.0, 1.0], [2.0, 2.0]])) is True
    assert ms.check_zero_volume_bounds(np.array([[0.0, 1.0], [2.0, 2.0]])) is False


def test_validate_n_starts_for_lhs_returns_unchanged() -> None:
    assert ms.validate_n_starts_for_lhs(3, 5) == 3  # warns but returns input
    assert ms.validate_n_starts_for_lhs(10, 4) == 10
    assert ms.validate_n_starts_for_lhs(100_000, 4) == 100_000  # large warn


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


_BOUNDS = np.array([[0.0, 1.0], [-2.0, 2.0], [10.0, 20.0]])


def test_generate_lhs_starts_within_bounds_and_reproducible() -> None:
    a = ms.generate_lhs_starts(_BOUNDS, n_starts=12, seed=3)
    b = ms.generate_lhs_starts(_BOUNDS, n_starts=12, seed=3)
    assert a.shape == (12, 3)
    np.testing.assert_array_equal(a, b)  # same seed -> identical
    assert np.all(a >= _BOUNDS[:, 0]) and np.all(a <= _BOUNDS[:, 1])


def test_generate_random_starts_within_bounds_and_reproducible() -> None:
    a = ms.generate_random_starts(_BOUNDS, n_starts=20, seed=5)
    b = ms.generate_random_starts(_BOUNDS, n_starts=20, seed=5)
    assert a.shape == (20, 3)
    np.testing.assert_array_equal(a, b)
    assert np.all(a >= _BOUNDS[:, 0]) and np.all(a <= _BOUNDS[:, 1])


# ---------------------------------------------------------------------------
# include_custom_starts
# ---------------------------------------------------------------------------


def test_include_custom_starts_none_or_empty() -> None:
    gen = np.zeros((3, 2))
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    assert ms.include_custom_starts(gen, None, bounds) is gen
    assert ms.include_custom_starts(gen, [], bounds) is gen


def test_include_custom_starts_prepends_valid() -> None:
    gen = np.full((3, 2), 0.5)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    combined = ms.include_custom_starts(gen, [[0.1, 0.2]], bounds)
    assert combined.shape == (4, 2)
    np.testing.assert_array_equal(combined[0], [0.1, 0.2])  # custom first


def test_include_custom_starts_wrong_dim_ignored() -> None:
    gen = np.zeros((3, 2))
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    out = ms.include_custom_starts(gen, [[0.1, 0.2, 0.3]], bounds)
    assert out is gen  # dimension mismatch -> ignored


def test_include_custom_starts_filters_out_of_bounds() -> None:
    gen = np.full((2, 2), 0.5)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    out = ms.include_custom_starts(gen, [[0.1, 0.2], [5.0, 5.0]], bounds)
    assert out.shape == (3, 2)  # only the valid custom prepended
    np.testing.assert_array_equal(out[0], [0.1, 0.2])


def test_include_custom_starts_all_out_of_bounds_returns_generated() -> None:
    gen = np.full((2, 2), 0.5)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    out = ms.include_custom_starts(gen, [[5.0, 5.0]], bounds)
    assert out is gen


# ---------------------------------------------------------------------------
# screening
# ---------------------------------------------------------------------------


def test_screen_starts_keeps_lowest_cost_sequential() -> None:
    starts = np.array([[3.0], [1.0], [2.0]])  # n_starts=3 -> sequential path
    filtered, costs = ms.screen_starts(lambda p: float(p[0]), starts, keep_fraction=0.5, min_keep=2)
    assert len(costs) == 3  # all costs returned
    assert len(filtered) == 2  # min_keep
    assert sorted(filtered[:, 0].tolist()) == [1.0, 2.0]  # two lowest kept


def test_screen_starts_parallel_path() -> None:
    starts = np.arange(6.0).reshape(6, 1)  # n_starts>=4 triggers thread pool
    filtered, costs = ms.screen_starts(
        lambda p: float(p[0]), starts, keep_fraction=0.5, n_workers=2
    )
    assert len(costs) == 6
    assert len(filtered) == 3
    assert filtered[0, 0] == 0.0  # lowest cost kept


# ---------------------------------------------------------------------------
# degeneracy detection
# ---------------------------------------------------------------------------


def test_detect_degeneracy_too_few_successful() -> None:
    results = [_single(1.0, success=True)]
    assert ms.detect_degeneracy(results) == (False, 1, None)


def test_detect_degeneracy_single_basin() -> None:
    r = [_single(1.0, success=True), _single(1.0, success=True)]
    detected, n_basins, labels = ms.detect_degeneracy(r)
    assert detected is False
    assert n_basins == 1
    assert labels is not None


def test_detect_degeneracy_multiple_basins() -> None:
    a = ms.SingleStartResult(0, np.zeros(2), np.array([1.0, 1.0]), 1.0, success=True)
    b = ms.SingleStartResult(1, np.zeros(2), np.array([5.0, 5.0]), 1.02, success=True)
    detected, n_basins, _ = ms.detect_degeneracy([a, b], chi_sq_threshold=0.1, param_threshold=0.2)
    assert detected is True
    assert n_basins == 2


# ---------------------------------------------------------------------------
# worker counting + sequential runner
# ---------------------------------------------------------------------------


def test_get_n_workers() -> None:
    assert ms.get_n_workers(ms.MultiStartConfig(n_workers=3), n_starts=10) == 3
    assert ms.get_n_workers(ms.MultiStartConfig(n_workers=3), n_starts=2) == 2  # capped
    auto = ms.get_n_workers(ms.MultiStartConfig(n_workers=0), n_starts=2)
    assert auto == 2  # capped to n_starts


def test_run_sequential_collects_and_handles_failure() -> None:
    starts = np.array([[1.0], [2.0]])

    def opt(idx: int, start: np.ndarray) -> ms.SingleStartResult:
        if idx == 1:
            raise RuntimeError("boom")
        return _single(float(start[0]), success=True)

    results = ms._run_sequential(opt, starts)
    assert len(results) == 2
    assert results[0].success is True
    assert results[1].success is False
    assert results[1].chi_squared == np.inf


# ---------------------------------------------------------------------------
# run_multistart_nlsq orchestration (sequential, in-process)
# ---------------------------------------------------------------------------


def _fit_factory(success: bool = True):
    def fit(data: dict[str, Any], p: np.ndarray) -> ms.SingleStartResult:
        chi = float(np.sum((np.asarray(p) - 0.5) ** 2))
        return _single(chi, success=success, cov=np.eye(len(p)))

    return fit


def test_run_multistart_sequential_happy_path() -> None:
    data = {"phi": np.zeros(8)}
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    cfg = ms.MultiStartConfig(enable=True, n_starts=4, seed=1, n_workers=1, use_screening=False)
    result = ms.run_multistart_nlsq(data, bounds, cfg, _fit_factory(success=True))
    assert result.strategy_used == "full"
    assert result.n_successful == 4
    assert len(result.all_results) == 4
    assert result.best.chi_squared == min(r.chi_squared for r in result.all_results)


def test_run_multistart_with_screening() -> None:
    data = {"phi": np.zeros(8)}
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    cfg = ms.MultiStartConfig(
        enable=True,
        n_starts=6,
        seed=1,
        n_workers=1,
        use_screening=True,
        screen_keep_fraction=0.5,
    )
    result = ms.run_multistart_nlsq(
        data,
        bounds,
        cfg,
        _fit_factory(success=True),
        cost_func=lambda p: float(np.sum(np.asarray(p) ** 2)),
    )
    assert result.screening_costs is not None
    assert result.strategy_used == "full"
    assert len(result.all_results) <= 6


def test_run_multistart_zero_volume_bounds_fallback() -> None:
    data = {"phi": np.zeros(4)}
    bounds = np.array([[1.0, 1.0], [2.0, 2.0]])  # zero volume
    cfg = ms.MultiStartConfig(enable=True, n_starts=4, n_workers=1, use_screening=False)
    result = ms.run_multistart_nlsq(data, bounds, cfg, _fit_factory(success=True))
    assert result.strategy_used == "single_start_fallback"
    assert len(result.all_results) == 1
    np.testing.assert_array_equal(result.best.initial_params, [1.0, 2.0])  # center


def test_run_multistart_all_failed() -> None:
    data = {"phi": np.zeros(4)}
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    cfg = ms.MultiStartConfig(enable=True, n_starts=3, n_workers=1, use_screening=False)
    result = ms.run_multistart_nlsq(data, bounds, cfg, _fit_factory(success=False))
    assert result.n_successful == 0
    assert result.best is result.all_results[0]
