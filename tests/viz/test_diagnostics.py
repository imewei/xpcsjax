"""Tests for compute_diagonal_overlay_stats."""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.viz.diagnostics import (
    DiagonalOverlayResult,
    compute_diagonal_overlay_stats,
)


def test_diagonal_overlay_shapes_match() -> None:
    n_phi, n = 3, 16
    rng = np.random.default_rng(0)
    c2_exp = rng.random((n_phi, n, n)) + 1.0
    c2_fit = c2_exp * 0.95
    result = compute_diagonal_overlay_stats(c2_exp, c2_fit, phi_index=1)
    assert isinstance(result, DiagonalOverlayResult)
    assert result.raw_diagonal.shape == (n,)
    assert result.fitted_diagonal.shape == (n,)
    assert result.phi_index == 1


def test_diagonal_overlay_rmse_matches_manual() -> None:
    n_phi, n = 2, 8
    c2_exp = np.ones((n_phi, n, n)) * 1.2
    c2_fit = np.ones((n_phi, n, n)) * 1.1
    result = compute_diagonal_overlay_stats(c2_exp, c2_fit, phi_index=0)
    assert result.fitted_rmse == pytest.approx(0.1)
    assert result.raw_variance == pytest.approx(0.0)
    assert result.fitted_variance == pytest.approx(0.0)


def test_diagonal_overlay_out_of_bounds_raises() -> None:
    n_phi, n = 3, 8
    c2_exp = np.ones((n_phi, n, n))
    c2_fit = np.ones((n_phi, n, n))
    with pytest.raises(IndexError):
        compute_diagonal_overlay_stats(c2_exp, c2_fit, phi_index=99)
