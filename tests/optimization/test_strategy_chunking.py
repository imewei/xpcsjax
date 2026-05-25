"""Tests for xpcsjax.optimization.nlsq.strategies.chunking.

All pure functions over numpy arrays: angle-distribution analysis, memory
estimation, adaptive chunk sizing, and angle-stratified reorganization. The
central invariants are conservation properties — stratification is a
*permutation* (no expansion, no loss): the sorted stratified phi equals the
sorted original, indices are a permutation of arange(n), and chunk sizes sum to
the point count. Decision logic (should_use_stratification) and diagnostics are
checked branch-by-branch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.strategies import chunking as ck

# create_angle_stratified_data is annotated with JAX Array params but accepts
# numpy at runtime (it copies via np.array internally). Wrap with Any-typed
# params so the type checker accepts the numpy fixtures used throughout.
_create_stratified = ck.create_angle_stratified_data


def _stratify(phi: Any, t1: Any, t2: Any, g2: Any, target_chunk_size: int = 100):
    return _create_stratified(phi, t1, t2, g2, target_chunk_size=target_chunk_size)


# ---------------------------------------------------------------------------
# analyze_angle_distribution
# ---------------------------------------------------------------------------


def test_analyze_angle_distribution_balanced() -> None:
    phi = np.array([0.0, 0.0, 45.0, 45.0, 90.0])
    stats = ck.analyze_angle_distribution(phi)
    assert stats.n_angles == 3
    assert stats.counts == {0.0: 2, 45.0: 2, 90.0: 1}
    assert stats.imbalance_ratio == pytest.approx(2.0)  # 2 / 1
    assert stats.is_balanced is True
    assert sum(stats.fractions.values()) == pytest.approx(1.0)
    assert stats.min_angle == 90.0  # fewest points
    assert stats.max_angle == 0.0  # most points (first max)


def test_analyze_angle_distribution_imbalanced() -> None:
    phi = np.concatenate([np.zeros(100), np.full(5, 90.0)])  # 20:1
    stats = ck.analyze_angle_distribution(phi)
    assert stats.imbalance_ratio == pytest.approx(20.0)
    assert stats.is_balanced is False


def test_analyze_angle_distribution_single_angle() -> None:
    stats = ck.analyze_angle_distribution(np.zeros(10))
    assert stats.n_angles == 1
    assert stats.imbalance_ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# memory estimation
# ---------------------------------------------------------------------------


def test_estimate_stratification_memory_full_copy() -> None:
    mem = ck.estimate_stratification_memory(10_000, n_features=4, use_index_based=False)
    expected_original = 10_000 * 4 * 8 / (1024**2)
    assert mem["original_memory_mb"] == pytest.approx(expected_original)
    assert mem["stratified_memory_mb"] == pytest.approx(expected_original)
    assert mem["peak_memory_mb"] == pytest.approx(2 * expected_original)
    assert mem["index_memory_mb"] == 0


def test_estimate_stratification_memory_index_based() -> None:
    mem = ck.estimate_stratification_memory(10_000, use_index_based=True)
    assert mem["stratified_memory_mb"] == 0  # only the index array is allocated
    assert mem["index_memory_mb"] > 0
    assert mem["peak_memory_mb"] == pytest.approx(
        mem["original_memory_mb"] + mem["index_memory_mb"]
    )


def test_estimate_stratification_memory_expansion() -> None:
    base = ck.estimate_stratification_memory(1000, estimated_expansion=1.0)
    expanded = ck.estimate_stratification_memory(1000, estimated_expansion=2.0)
    assert expanded["stratified_memory_mb"] == pytest.approx(
        2 * base["stratified_memory_mb"]
    )


def test_estimate_nlsq_optimization_memory_jacobian_dominant() -> None:
    mem = ck.estimate_nlsq_optimization_memory(100_000, n_params=53)
    expected_jac = 100_000 * 53 * 8 / (1024**2)
    assert mem["jacobian_mb"] == pytest.approx(expected_jac)
    assert mem["peak_gb"] > 0
    assert set(mem) >= {"data_mb", "jacobian_mb", "total_mb", "peak_gb", "is_safe"}


# ---------------------------------------------------------------------------
# calculate_adaptive_chunk_size
# ---------------------------------------------------------------------------


def test_adaptive_chunk_size_clamped_to_max() -> None:
    # Small dataset, few params, plenty of memory -> clamps to max_chunk_size.
    size = ck.calculate_adaptive_chunk_size(
        total_points=1_000_000, n_params=9, n_angles=3, available_memory_gb=64.0
    )
    assert size == 500_000


def test_adaptive_chunk_size_memory_constrained() -> None:
    size = ck.calculate_adaptive_chunk_size(
        total_points=23_000_000, n_params=53, n_angles=23, available_memory_gb=1.0
    )
    assert 10_000 <= size <= 500_000  # within clamps, memory-limited


def test_adaptive_chunk_size_clamped_to_min() -> None:
    size = ck.calculate_adaptive_chunk_size(
        total_points=10_000_000, n_params=10_000, n_angles=5, available_memory_gb=0.1
    )
    assert size == 10_000  # clamped up to min


def test_adaptive_chunk_size_zero_angles() -> None:
    size = ck.calculate_adaptive_chunk_size(
        total_points=1_000_000, n_params=9, n_angles=0, available_memory_gb=32.0
    )
    assert 10_000 <= size <= 500_000


# ---------------------------------------------------------------------------
# create_angle_stratified_data / _indices — conservation invariants
# ---------------------------------------------------------------------------


def _balanced_dataset(n_per_angle: int = 100):
    angles = [0.0, 45.0, 90.0]
    phi = np.repeat(angles, n_per_angle)
    n = len(phi)
    t1 = np.arange(n, dtype=np.float64)
    t2 = np.arange(n, dtype=np.float64) + 0.5
    g2 = np.linspace(1.0, 2.0, n)
    return phi, t1, t2, g2


def test_stratified_data_is_permutation() -> None:
    phi, t1, t2, g2 = _balanced_dataset(100)  # 300 points, 3 angles
    phi_s, t1_s, t2_s, g2_s, chunk_sizes = _stratify(
        phi, t1, t2, g2, target_chunk_size=100
    )
    phi_s = np.asarray(phi_s)
    assert len(phi_s) == 300  # no expansion
    np.testing.assert_array_equal(np.sort(phi_s), np.sort(phi))  # permutation
    assert sum(chunk_sizes) == 300
    # Each chunk should contain all 3 angles (balanced -> stratified).
    start = 0
    for size in chunk_sizes:
        chunk = phi_s[start : start + size]
        assert len(np.unique(chunk)) == 3
        start += size


def test_stratified_data_single_angle_passthrough() -> None:
    phi = np.zeros(50)
    t1 = np.arange(50.0)
    out = _stratify(phi, t1, t1, t1, target_chunk_size=10)
    assert out[4] == [50]  # chunk_sizes = [n_points], no reorganization


def test_stratified_indices_is_permutation() -> None:
    phi, *_ = _balanced_dataset(100)
    indices, chunk_sizes = ck.create_angle_stratified_indices(phi, target_chunk_size=100)
    np.testing.assert_array_equal(np.sort(indices), np.arange(300))  # permutation
    assert sum(chunk_sizes) == 300


def test_stratified_indices_single_angle() -> None:
    indices, chunk_sizes = ck.create_angle_stratified_indices(np.zeros(20))
    np.testing.assert_array_equal(indices, np.arange(20))
    assert chunk_sizes == [20]


# ---------------------------------------------------------------------------
# iterator
# ---------------------------------------------------------------------------


def test_stratified_index_iterator() -> None:
    it = ck.StratifiedIndexIterator(np.arange(10), [3, 3, 4])
    chunks = list(it)
    assert len(it) == 3
    assert [len(c) for c in chunks] == [3, 3, 4]
    np.testing.assert_array_equal(np.concatenate(chunks), np.arange(10))


def test_get_stratified_chunk_iterator() -> None:
    phi, *_ = _balanced_dataset(50)  # 150 points
    it = ck.get_stratified_chunk_iterator(phi, target_chunk_size=50)
    assert isinstance(it, ck.StratifiedIndexIterator)
    assert sum(len(c) for c in it) == 150


# ---------------------------------------------------------------------------
# should_use_stratification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_points", "n_angles", "per_angle", "imbalance", "expected"),
    [
        (50_000, 3, True, 2.0, False),  # too small
        (1_000_000, 3, False, 2.0, False),  # per-angle disabled
        (1_000_000, 1, True, 1.0, False),  # single angle
        (1_000_000, 3, True, 8.0, False),  # too imbalanced
        (1_000_000, 3, True, 2.0, True),  # all conditions met
    ],
)
def test_should_use_stratification(
    n_points: int, n_angles: int, per_angle: bool, imbalance: float, expected: bool
) -> None:
    should, reason = ck.should_use_stratification(n_points, n_angles, per_angle, imbalance)
    assert should is expected
    assert isinstance(reason, str) and reason


# ---------------------------------------------------------------------------
# diagnostics + report
# ---------------------------------------------------------------------------


def test_compute_diagnostics_with_chunk_sizes() -> None:
    phi, t1, t2, g2 = _balanced_dataset(100)
    phi_s, *_rest, chunk_sizes = _stratify(
        phi, t1, t2, g2, target_chunk_size=100
    )
    diag = ck.compute_stratification_diagnostics(
        phi, np.asarray(phi_s), execution_time_ms=10.0, chunk_sizes=chunk_sizes
    )
    assert diag.n_chunks == len(chunk_sizes)
    assert set(diag.chunk_balance) == {"mean", "std", "min", "max", "cv"}
    assert diag.angle_coverage["perfect_coverage_chunks"] == diag.n_chunks  # all balanced
    assert diag.throughput_points_per_sec > 0


def test_compute_diagnostics_fallback_slicing() -> None:
    phi, t1, t2, g2 = _balanced_dataset(100)
    phi_s, *_rest = _stratify(phi, t1, t2, g2, target_chunk_size=100)[:4]
    diag = ck.compute_stratification_diagnostics(
        phi, np.asarray(phi_s), execution_time_ms=5.0, target_chunk_size=100
    )
    assert diag.n_chunks == 3  # ceil(300 / 100)
    assert diag.execution_time_ms == 5.0


def test_format_diagnostics_report() -> None:
    phi, t1, t2, g2 = _balanced_dataset(50)
    phi_s, *_rest, chunk_sizes = _stratify(
        phi, t1, t2, g2, target_chunk_size=50
    )
    diag = ck.compute_stratification_diagnostics(
        phi, np.asarray(phi_s), execution_time_ms=2.0, chunk_sizes=chunk_sizes
    )
    report = ck.format_diagnostics_report(diag)
    assert "STRATIFICATION DIAGNOSTICS REPORT" in report
    assert "Chunk Balance:" in report
    assert "Angle Coverage:" in report
