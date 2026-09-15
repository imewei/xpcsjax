"""Unit tests for the shared covariance post-processing (nlsq/covariance.py)."""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.covariance import (
    finalize_covariance,
    gauss_newton_covariance,
    rescale_covariance_dof,
    robust_scale_residual_jacobian,
)


def test_finalize_rejects_inf_zero_negative_and_missing():
    ok = np.diag([1.0, 2.0])
    cov, unc, flag = finalize_covariance(ok, 2)
    assert flag is False
    np.testing.assert_array_equal(cov, ok)
    np.testing.assert_allclose(unc, np.sqrt([1.0, 2.0]))

    for bad in (
        np.full((2, 2), np.inf),  # nlsq singular marker
        np.diag([1.0, 0.0]),  # pinv null-space "infinite precision"
        np.diag([1.0, -1e-3]),  # non-PD Hessian
        np.array([[1.0, np.nan], [np.nan, 1.0]]),
        np.eye(3),  # wrong shape
        None,
    ):
        cov, unc, flag = finalize_covariance(bad, 2)
        assert flag is True
        assert cov.shape == (2, 2) and np.all(np.isnan(cov))
        assert unc.shape == (2,) and np.all(np.isnan(unc))


@pytest.mark.parametrize("loss", ["linear", "huber", "soft_l1", "cauchy", "arctan"])
def test_robust_scaling_matches_nlsq(loss):
    """Host mirror == nlsq's own CommonJIT.scale_for_robust_loss_function."""
    import jax.numpy as jnp
    from nlsq.common_jax import CommonJIT
    from nlsq.core.loss_functions import LossFunctionsJIT

    rng = np.random.default_rng(3)
    f = rng.normal(0.0, 1.5, size=40)  # O(1) residuals: scaling is NOT identity
    J = rng.normal(size=(40, 3))
    f_s, J_s, cost = robust_scale_residual_jacobian(f, J, loss=loss, f_scale=1.0)
    if loss == "linear":
        np.testing.assert_array_equal(f_s, f)
        np.testing.assert_array_equal(J_s, J)
        assert cost == pytest.approx(float(np.sum(f * f)))
        return
    lf = LossFunctionsJIT()
    rho = lf.construct_single_loss_function(lf.IMPLEMENTED_LOSSES[loss])(jnp.asarray(f), 1.0)
    J_ref, f_ref = CommonJIT().scale_for_robust_loss_function(jnp.asarray(J), jnp.asarray(f), rho)
    np.testing.assert_allclose(f_s, np.asarray(f_ref), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(J_s, np.asarray(J_ref), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(cost, float(np.sum(np.asarray(rho[0]))), rtol=1e-12)


def test_gauss_newton_covariance_matches_nlsq_curve_fit_estimator():
    """s²(J_sᵀJ_s)⁻¹ with the robust-scaled J equals nlsq's curve_fit pcov."""
    from nlsq import CurveFit

    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 4.0, 60)
    y = 2.0 * np.exp(-0.7 * x) + 0.3 + rng.normal(0.0, 0.3, size=x.size)

    def model(x, a, b, c):
        import jax.numpy as jnp

        return a * jnp.exp(-b * x) + c

    popt, pcov = CurveFit().curve_fit(model, x, y, p0=[1.0, 1.0, 0.0], loss="soft_l1")
    popt = np.asarray(popt)
    a, b, c = popt
    f = y - (a * np.exp(-b * x) + c)  # residual sign convention does not matter
    J = np.column_stack([-np.exp(-b * x), a * x * np.exp(-b * x), -np.ones_like(x)])
    ours = gauss_newton_covariance(f, J, n_valid=x.size, n_params=3, loss="soft_l1")
    np.testing.assert_allclose(ours, np.asarray(pcov), rtol=2e-3)


def test_gauss_newton_covariance_is_strict():
    f = np.ones(5)
    J = np.column_stack([np.ones(5), np.ones(5)])  # rank 1
    with pytest.raises(np.linalg.LinAlgError):
        gauss_newton_covariance(f, J, n_valid=5, n_params=2)


def test_rescale_covariance_dof():
    pcov = np.eye(2)
    np.testing.assert_allclose(
        rescale_covariance_dof(pcov, n_rows_solver=1083, n_valid=1026, n_params=16),
        pcov * (1083 - 16) / (1026 - 16),
    )
