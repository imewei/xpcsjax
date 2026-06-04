"""Coverage for heterodyne data-prep pure functions (audit finding #16).

Exercises the previously-uncovered ``raise ValueError`` paths and the
flatten/unflatten round-trip — all deterministic, no optimizer execution.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.heterodyne_data_prep import (
    compute_degrees_of_freedom,
    compute_weights,
    far_lag_noise_variance,
    flatten_upper_triangle,
    noise_normalized_reduced_chi2,
    unflatten_upper_triangle,
)


def test_flatten_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="square matrix"):
        flatten_upper_triangle(np.zeros((3, 4)))


@pytest.mark.parametrize("include_diagonal", [True, False])
def test_flatten_unflatten_round_trip(include_diagonal: bool) -> None:
    n = 4
    mat = np.arange(n * n, dtype=float).reshape(n, n)
    sym = mat + mat.T  # symmetric so the round-trip is well-defined
    flat = flatten_upper_triangle(sym, include_diagonal=include_diagonal)
    recon = unflatten_upper_triangle(flat, n, include_diagonal=include_diagonal)
    iu = np.triu_indices(n, k=0 if include_diagonal else 1)
    assert np.allclose(recon[iu], sym[iu])
    assert np.allclose(recon, recon.T)  # reconstruction is symmetric


def test_unflatten_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="Expected"):
        unflatten_upper_triangle(np.zeros(5), n=4, include_diagonal=True)


def test_compute_weights_inverse_variance_requires_sigma() -> None:
    with pytest.raises(ValueError, match="sigma required"):
        compute_weights(np.ones((3, 3)), method="inverse_variance", sigma=None)


def test_compute_weights_inverse_variance_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="doesn't match"):
        compute_weights(np.ones((3, 3)), method="inverse_variance", sigma=np.ones((2, 2)))


def test_compute_weights_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown weight method"):
        compute_weights(np.ones((3, 3)), method="bogus")


def test_compute_weights_exclude_diagonal_zeros_diagonal() -> None:
    w = compute_weights(np.ones((3, 3)), method="uniform", exclude_diagonal=True)
    assert np.allclose(np.diag(w), 0.0)
    assert w[0, 1] == 1.0


def test_degrees_of_freedom_normal() -> None:
    assert compute_degrees_of_freedom(n_data=100, n_params=7) == 93


def test_degrees_of_freedom_underdetermined_floors_at_one() -> None:
    # n_data <= n_params hits the warning branch and floors dof at 1.
    assert compute_degrees_of_freedom(n_data=5, n_params=7) == 1
    assert compute_degrees_of_freedom(n_data=7, n_params=7) == 1


def test_far_lag_noise_variance_batched_matches_pooled() -> None:
    # 3-D (n_phi, n_time, n_time): far-lag entries pooled across angles.
    rng = np.random.default_rng(0)
    c2 = rng.normal(1.0, 0.05, size=(3, 8, 8))
    var = far_lag_noise_variance(c2)
    # Reproduce the far-lag mask (|i-j| >= n_time//2) and compare.
    n = 8
    idx = np.arange(n)
    mask = np.abs(idx[:, None] - idx[None, :]) >= n // 2
    expected = float(np.var(c2[:, mask].ravel()))
    assert var == pytest.approx(expected)


def test_far_lag_noise_variance_degenerate_is_zero() -> None:
    # A perfectly flat far-lag tail -> zero variance (triggers MSE fallback).
    assert far_lag_noise_variance(np.ones((4, 4))) == 0.0


def test_noise_normalized_reduced_chi2_targets_one() -> None:
    # When SSR/dof equals sigma2_noise, the normalised chi2 is exactly 1.
    n_time = 8
    idx = np.arange(n_time)
    mask = np.abs(idx[:, None] - idx[None, :]) >= n_time // 2
    # Build C2 whose far-lag variance is a known value.
    c2 = np.ones((n_time, n_time))
    far_vals = np.array([0.9, 1.1] * (int(mask.sum()) // 2 + 1))[: int(mask.sum())]
    c2[mask] = far_vals
    sigma2 = far_lag_noise_variance(c2)
    assert sigma2 > 0
    n_valid, n_params = 50, 6
    dof = n_valid - n_params
    ssr = sigma2 * dof  # arrange chi2_red == 1.0
    chi2_red = noise_normalized_reduced_chi2(ssr, c2, n_valid, n_params)
    assert chi2_red == pytest.approx(1.0)


def test_noise_normalized_reduced_chi2_mse_fallback() -> None:
    # Degenerate noise -> falls back to plain MSE = SSR / dof.
    c2 = np.ones((4, 4))  # zero far-lag variance
    ssr, n_valid, n_params = 8.0, 10, 2
    out = noise_normalized_reduced_chi2(ssr, c2, n_valid, n_params)
    assert out == pytest.approx(ssr / (n_valid - n_params))
