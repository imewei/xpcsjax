"""Covariance recovery when nlsq reports a rank-deficient fit as all-``inf``.

nlsq >= 0.8 returns an all-``inf`` covariance when it cannot trust the
pseudo-inverse it builds for a rank-deficient Jacobian. The heterodyne fraction
``f0 * exp(f1 * (t - f2)) + f3`` determines only ``f0 * exp(-f1 * f2)``, so that
happens routinely to ``two_component`` fits.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.covariance import rank_deficient_covariance
from xpcsjax.optimization.nlsq.heterodyne_result_builder import (
    build_result_from_nlsq,
    label_unidentified_fraction_parameters,
)


def _exact_pinv_case() -> tuple[np.ndarray, np.ndarray]:
    """Duplicate columns beside a column 1e10 smaller, and the exact answer."""
    u = np.array([1.0, 2.0, 3.0, 4.0])
    v = np.array([1.0, -1.0, -1.0, 1.0])  # orthogonal to u
    J = np.column_stack([1e5 * u, 1e-5 * v, 1e5 * u])
    pair = 1.0 / (4.0 * np.sum((1e5 * u) ** 2))
    X = np.array([[pair, 0.0, pair], [0.0, 0.0, 0.0], [pair, 0.0, pair]])
    X[1, 1] = 1.0 / np.sum((1e-5 * v) ** 2)
    return J, X


def test_matches_exact_pseudo_inverse_across_extreme_column_scales() -> None:
    J, expected = _exact_pinv_case()
    # n=4, p=3, cost=0.5 -> nlsq's scale 2 * cost / (n - p) is exactly 1
    pcov = rank_deficient_covariance(J, cost=0.5, n_data=4, pair=(0, 2))
    sd = np.sqrt(np.diag(expected))
    np.testing.assert_allclose((pcov - expected) / np.outer(sd, sd), 0.0, atol=1e-12)


def test_equals_numpy_pinv_when_well_scaled() -> None:
    rng = np.random.default_rng(0)
    B = rng.normal(size=(60, 4))
    J = np.column_stack([B, -0.5 * B[:, 2]])
    pcov = rank_deficient_covariance(J, cost=3.0, n_data=60, pair=(2, 4))
    expected = np.linalg.pinv(J.T @ J, rcond=1e-12) * (2.0 * 3.0 / (60 - 5))
    np.testing.assert_allclose(pcov, expected, rtol=1e-9, atol=1e-12)


def _third_column_involved(weight: float, scale: float) -> np.ndarray:
    """Columns ``(u, scale v, scale v + weight u)``: all three are inseparable."""
    u, v = np.eye(4)[:2]
    return np.column_stack([u, scale * v, scale * v + weight * u])


@pytest.mark.parametrize(
    ("J", "pair"),
    [
        # another parameter enters the dependency, visibly or at round-off level
        (_third_column_involved(5.0, 1e9), (1, 2)),
        (_third_column_involved(5.0, 1e14), (1, 2)),
        (_third_column_involved(2.0, 1e14), (1, 2)),
        (_third_column_involved(0.05, 1e14), (1, 2)),
        (_third_column_involved(1e14 * (np.exp(np.log(1e100)) / 1e100 - 1), 1e14), (1, 2)),
        (_third_column_involved(0.5, 1.0), (1, 2)),
        # a clean pair beside a column 1e20 smaller: round-off could couple them
        (np.column_stack([1e10 * np.eye(4)[0], 1e-10 * np.eye(4)[1], 1e10 * np.eye(4)[0]]), (0, 2)),
        # two dependencies, and a dependency the caller does not know about
        (np.eye(6, 2) @ np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]), (0, 2)),
        (np.column_stack([np.eye(4)[0], np.eye(4)[1], np.eye(4)[0]]), None),
        (np.zeros((6, 2)), (0, 1)),  # rank zero
    ],
)
def test_reports_unknown_unless_the_pair_is_the_only_dependency(J, pair) -> None:
    pcov = rank_deficient_covariance(J, cost=0.5, n_data=len(J), pair=pair)
    assert pcov is not None and np.isnan(pcov).all()


@pytest.mark.parametrize(
    ("J", "cost", "n_data"),
    [
        (None, 1.0, 10),
        (np.eye(4, 3), 1.0, 4),  # full rank: an all-inf pcov has another cause
        (np.array([[1.0, 1.0], [np.nan, 1.0], [1.0, 1.0]]), 1.0, 3),
        (np.ones((3, 2)), np.nan, 3),
        (np.ones((2, 2)), 1.0, 2),  # no degrees of freedom
        (np.zeros((4, 0)), 1.0, 4),  # no parameters
        (np.zeros((0, 3)), 1.0, 5),  # no rows
    ],
)
def test_declines_when_the_jacobian_says_nothing(J, cost, n_data) -> None:
    assert rank_deficient_covariance(J, cost=cost, n_data=n_data, pair=(0, 1)) is None


def _solver_result(J: np.ndarray, **extra) -> dict:
    p = J.shape[1]
    return {"x": np.ones(p), "pcov": np.full((p, p), np.inf), "fun": np.zeros(len(J)), **extra}


def test_result_builder_recovers_all_inf_covariance() -> None:
    J, expected = _exact_pinv_case()
    result = build_result_from_nlsq(_solver_result(J, jac=J, cost=0.5), ["f0", "b", "f2"], 4)
    assert result.metadata["covariance_is_placeholder"] is False
    np.testing.assert_allclose(result.uncertainties, np.sqrt(np.diag(expected)), rtol=1e-12)


def test_result_builder_replaces_a_finite_covariance_on_a_rank_deficient_jacobian() -> None:
    # nlsq can accept a rank-deficient covariance whose degenerate entries are wrong
    J, expected = _exact_pinv_case()
    wrong = _solver_result(J, jac=J, cost=0.5) | {"pcov": 100.0 * expected}
    result = build_result_from_nlsq(wrong, ["f0", "b", "f2"], 4)
    np.testing.assert_allclose(result.uncertainties, np.sqrt(np.diag(expected)), rtol=1e-12)


def test_result_builder_keeps_the_solver_covariance_when_full_rank() -> None:
    J = np.eye(5, 3)
    solver = _solver_result(J, jac=J, cost=0.5) | {"pcov": 7.0 * np.eye(3)}
    result = build_result_from_nlsq(solver, ["a", "b", "c"], 5)
    np.testing.assert_array_equal(result.covariance, 7.0 * np.eye(3))


@pytest.mark.parametrize("weight", [5.0, 2.0])
def test_result_builder_never_hides_a_third_inseparable_parameter(weight) -> None:
    J = _third_column_involved(weight, 1e14)
    solve = _solver_result(J, pcov=np.eye(3), jac=J, cost=0.5)
    result = build_result_from_nlsq(solve, ["D0_sample", "f0", "f2"], 4)
    assert np.isnan(result.uncertainties).all()
    assert result.metadata["covariance_is_placeholder"] is True


@pytest.mark.parametrize("solver_pcov", ["rejected", "accepted"])
@pytest.mark.parametrize(
    ("names", "known"),
    [(["f0", "b", "f2"], True), (["x", "y", "z"], False)],
    ids=["labelled-pair", "other-parameters"],
)
def test_result_builder_reports_only_the_labelled_degeneracy(solver_pcov, names, known) -> None:
    u, v = np.eye(4)[:2]
    J = np.column_stack([u, v, u])  # first and last columns are inseparable
    pcov = np.eye(3) if solver_pcov == "accepted" else np.full((3, 3), np.inf)
    solve = _solver_result(J, pcov=pcov, jac=J, cost=0.5)

    result = build_result_from_nlsq(solve, names, 4)

    if known:
        # exact pinv(JᵀJ); numpy's own carries round-off in the zeros on some CPUs
        expected = np.array([[0.25, 0.0, 0.25], [0.0, 1.0, 0.0], [0.25, 0.0, 0.25]])
        np.testing.assert_allclose(result.covariance, expected, rtol=1e-7, atol=1e-12)
        assert result.metadata["covariance_is_placeholder"] is False
    else:
        assert np.all(np.isnan(result.covariance))
        assert np.all(np.isnan(result.uncertainties))
        assert result.metadata["covariance_is_placeholder"] is True


@pytest.mark.parametrize(
    ("f2_sigma", "expected"),
    [(3e-4, ["f0", "f2"]), (np.nan, None), (0.0, None)],  # free, unknown, fixed
)
def test_free_fraction_pair_is_labelled_unidentified(f2_sigma, expected) -> None:
    result = SimpleNamespace(
        uncertainties=np.array([0.1, 0.03, 0.001, f2_sigma]),
        nlsq_diagnostics={"parameter_names": ["contrast_avg", "f0", "f1", "f2"]},
    )
    labelled = label_unidentified_fraction_parameters(result)
    assert labelled.nlsq_diagnostics.get("covariance_unidentified_parameters") == expected


@pytest.mark.parametrize("names", [["f0", "f2"], ["a", "b", "f0", "f2"]])
def test_result_builder_ignores_a_jacobian_that_does_not_match_the_names(names) -> None:
    J, _ = _exact_pinv_case()  # three columns
    solve = _solver_result(J, jac=J, cost=0.5) | {"x": np.ones(len(names))}
    solve["pcov"] = np.eye(len(names))
    result = build_result_from_nlsq(solve, names, 4)
    np.testing.assert_array_equal(result.covariance, np.eye(len(names)))


def test_result_builder_keeps_placeholder_without_a_jacobian() -> None:
    J, _ = _exact_pinv_case()
    result = build_result_from_nlsq(_solver_result(J), ["a", "b", "c"], 4)
    assert result.metadata["covariance_is_placeholder"] is True
    assert np.isnan(result.uncertainties).all()
