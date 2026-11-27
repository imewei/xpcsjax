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
    label_unidentified_parameters,
    record_unidentified,
)


def _exact_pinv_case() -> tuple[np.ndarray, np.ndarray]:
    """nlsq's own rejected Jacobian (duplicate columns, 1e20 scale ratio) and truth."""
    u = np.array([1.0, 2.0, 3.0, 4.0])
    v = np.array([1.0, -1.0, -1.0, 1.0])  # orthogonal to u
    J = np.column_stack([1e10 * u, 1e-10 * v, 1e10 * u])
    pair = 1.0 / (4.0 * np.sum((1e10 * u) ** 2))
    X = np.array([[pair, 0.0, pair], [0.0, 0.0, 0.0], [pair, 0.0, pair]])
    X[1, 1] = 1.0 / np.sum((1e-10 * v) ** 2)
    return J, X


def test_matches_exact_pseudo_inverse_across_extreme_column_scales() -> None:
    J, expected = _exact_pinv_case()
    # n=4, p=3, cost=0.5 -> nlsq's scale 2 * cost / (n - p) is exactly 1
    pcov, unidentified = rank_deficient_covariance(J, cost=0.5, n_data=4)
    np.testing.assert_array_equal(unidentified, [True, False, True])
    sd = np.sqrt(np.diag(expected))
    np.testing.assert_allclose((pcov - expected) / np.outer(sd, sd), 0.0, atol=1e-12)


def test_equals_numpy_pinv_when_well_scaled() -> None:
    rng = np.random.default_rng(0)
    B = rng.normal(size=(60, 4))
    # one dependency spanning two columns, one column outside it untouched
    J = np.column_stack([B, 2.0 * B[:, 0] - 0.5 * B[:, 2]])
    pcov, unidentified = rank_deficient_covariance(J, cost=3.0, n_data=60)
    np.testing.assert_array_equal(unidentified, [True, False, True, False, True])
    expected = np.linalg.pinv(J.T @ J, rcond=1e-12) * (2.0 * 3.0 / (60 - 5))
    np.testing.assert_allclose(pcov, expected, rtol=1e-9, atol=1e-12)


def test_keeps_a_small_coefficient_that_is_large_in_parameter_units() -> None:
    u, v = np.eye(4)[:2]
    # null vector (-5, -1, 1): 5 in parameter units, 5e-9 between unit-norm columns
    J = np.column_stack([u, 1e9 * v, 1e9 * v + 5.0 * u])
    pcov, unidentified = rank_deficient_covariance(J, cost=0.5, n_data=4)
    P = np.linalg.pinv(J, rcond=1e-12)
    np.testing.assert_array_equal(unidentified, [True, True, True])
    np.testing.assert_allclose(pcov, P @ P.T, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize(
    ("J", "cost", "n_data"),
    [
        (None, 1.0, 10),
        (np.eye(4, 3), 1.0, 4),  # full rank: an all-inf pcov has another cause
        (np.zeros((6, 2)), 1.0, 6),  # rank zero
        (np.array([[1.0, 1.0], [np.nan, 1.0], [1.0, 1.0]]), 1.0, 3),
        (np.ones((3, 2)), np.nan, 3),
        (np.ones((2, 2)), 1.0, 2),  # no degrees of freedom
        (np.zeros((4, 0)), 1.0, 4),  # no parameters
    ],
)
def test_declines_when_covariance_is_not_recoverable(J, cost, n_data) -> None:
    assert rank_deficient_covariance(J, cost=cost, n_data=n_data) is None


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


def test_result_builder_names_every_degenerate_parameter() -> None:
    u, v = np.eye(4)[:2]
    J = np.column_stack([u, v, u + v])
    result = build_result_from_nlsq(_solver_result(J, jac=J, cost=0.5), ["x", "y", "z"], 4)
    assert result.metadata["covariance_unidentified_parameters"] == ["x", "y", "z"]


def test_label_lists_recorded_degeneracies_that_report_a_sigma() -> None:
    result = SimpleNamespace(
        uncertainties=np.array([0.1, np.nan, 0.2, 0.3, 0.4]),
        nlsq_diagnostics={"parameter_names": ["D0", "alpha", "f0", "f1", "f2"]},
    )
    record_unidentified(
        result.nlsq_diagnostics,
        SimpleNamespace(metadata={"covariance_unidentified_parameters": ["alpha", "D0"]}),
        None,
    )
    labelled = label_unidentified_parameters(result)
    assert labelled.nlsq_diagnostics["covariance_unidentified_parameters"] == ["D0", "f0", "f2"]


@pytest.mark.parametrize(
    ("f2_sigma", "expected"),
    [(3e-4, ["f0", "f2"]), (np.nan, None), (0.0, None)],  # free, unknown, fixed
)
def test_free_fraction_pair_is_labelled_unidentified(f2_sigma, expected) -> None:
    result = SimpleNamespace(
        uncertainties=np.array([0.1, 0.03, 0.001, f2_sigma]),
        nlsq_diagnostics={"parameter_names": ["contrast_avg", "f0", "f1", "f2"]},
    )
    labelled = label_unidentified_parameters(result)
    assert labelled.nlsq_diagnostics.get("covariance_unidentified_parameters") == expected


def test_result_builder_keeps_placeholder_without_a_jacobian() -> None:
    J, _ = _exact_pinv_case()
    result = build_result_from_nlsq(_solver_result(J), ["a", "b", "c"], 4)
    assert result.metadata["covariance_is_placeholder"] is True
    assert np.isnan(result.uncertainties).all()
