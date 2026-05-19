"""Smoke tests for the Jacobian-stats utility module.

``xpcsjax/optimization/nlsq/jacobian.py`` provides Jacobian diagnostics —
J^T J for Hessian approximation, condition numbers for ill-conditioning
warnings, column norms for parameter sensitivity, and gradient noise
estimates. The functions are pure JAX/numpy and easily fixture-able with a
polynomial residual.

These tests pin the contract: each function returns sensible values for a
well-conditioned polynomial fit, and degrades gracefully (returns None or
empty dict) when the inputs make the math undefined.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from xpcsjax.optimization.nlsq.jacobian import (
    analyze_parameter_sensitivity,
    compute_jacobian_condition_number,
    compute_jacobian_stats,
    estimate_gradient_noise,
)


# Toy residual: r_i = y_i - (a + b*x_i + c*x_i^2). Three parameters, 20
# residuals — well-conditioned, easy to differentiate, small enough that
# every test runs in under a second.
def _polynomial_residual(xdata: np.ndarray, a: float, b: float, c: float) -> jnp.ndarray:
    x = jnp.asarray(xdata)
    pred = a + b * x + c * x**2
    # Synthetic data: true model with (a,b,c) = (1.0, 2.0, 3.0). Test residual
    # is evaluated at perturbed params, giving non-trivial Jacobian.
    truth = 1.0 + 2.0 * x + 3.0 * x**2
    return pred - truth


@pytest.fixture
def fixture_data() -> tuple[np.ndarray, np.ndarray]:
    """20-point polynomial fixture: xdata + initial params at the truth."""
    xdata = np.linspace(0.0, 1.0, 20, dtype=np.float64)
    params = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    return xdata, params


def test_compute_jacobian_stats_returns_jtj_and_norms(
    fixture_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Happy path: well-conditioned polynomial returns a (3,3) J^T J and
    a length-3 column-norm vector. Both are finite."""
    xdata, params = fixture_data
    jtj, col_norms = compute_jacobian_stats(
        _polynomial_residual, xdata, params, scaling_factor=1.0
    )

    assert jtj is not None and col_norms is not None
    assert jtj.shape == (3, 3)
    assert col_norms.shape == (3,)
    assert np.all(np.isfinite(jtj))
    assert np.all(np.isfinite(col_norms))
    # J^T J is symmetric positive semi-definite.
    np.testing.assert_allclose(jtj, jtj.T, atol=1e-12)


def test_compute_jacobian_stats_handles_broken_residual() -> None:
    """A residual function that raises must yield (None, None) rather than
    propagating the exception. The diagnostics path is best-effort, not
    fatal — a None return lets the caller fall through to a default."""
    def broken(xdata, *params):  # noqa: ARG001
        raise RuntimeError("intentional")

    xdata = np.linspace(0.0, 1.0, 5)
    params = np.array([1.0, 2.0, 3.0])

    jtj, col_norms = compute_jacobian_stats(broken, xdata, params, 1.0)
    assert jtj is None
    assert col_norms is None


def test_compute_jacobian_condition_number_is_finite(
    fixture_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Well-conditioned polynomial fit has a finite, modest cond(J)."""
    xdata, params = fixture_data
    cond = compute_jacobian_condition_number(_polynomial_residual, xdata, params)

    assert cond is not None
    assert np.isfinite(cond)
    assert cond > 0  # Always positive for a real matrix


def test_analyze_parameter_sensitivity_normalized_to_unit(
    fixture_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Sensitivities are normalized so the most-influential parameter == 1.0.
    Catches a regression where the normalization is dropped (max != 1.0)
    or inverted (smallest = 1.0)."""
    xdata, params = fixture_data
    sensitivities = analyze_parameter_sensitivity(
        _polynomial_residual, xdata, params, ["a", "b", "c"]
    )

    assert set(sensitivities.keys()) == {"a", "b", "c"}
    assert max(sensitivities.values()) == pytest.approx(1.0)
    assert all(0.0 <= v <= 1.0 for v in sensitivities.values())


def test_estimate_gradient_noise_finite_for_smooth_residual(
    fixture_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Smooth polynomial → low but finite gradient noise across
    perturbations. The exact value depends on float64 round-off; we just
    pin that the diagnostic returns a finite number rather than crashing."""
    xdata, params = fixture_data
    noise = estimate_gradient_noise(
        _polynomial_residual, xdata, params, n_samples=4, perturbation=1e-6
    )

    assert noise is not None
    assert np.isfinite(noise)
    assert noise >= 0
