"""R2 (status-grading parity): laminar stratified-LS must grade a max_nfev-limited
solve with a good reduced chi-squared as ``max_iter`` (graded on its real chi^2),
not a blanket ``failed`` — mirroring ``heterodyne_result_builder`` (the heterodyne
path already does this; laminar's ``_create_fit_result`` did not).

These tests pin the centralized grading in ``NLSQWrapper._create_fit_result`` so
both the stratified-LS and hybrid-streaming call sites inherit it.

The relabel is numerics-safe: parameters / chi^2 / covariance are unchanged; only
``convergence_status`` is upgraded from ``failed`` to ``max_iter`` when the solver
hit its function-evaluation budget yet produced a finite reduced chi^2.
"""

from __future__ import annotations

import numpy as np

from xpcsjax.optimization.nlsq.wrapper import NLSQWrapper


def _wrapper() -> NLSQWrapper:
    return NLSQWrapper(["p0", "p1"], max_retries=0)


def _good_chi2_inputs() -> dict:
    # 100 data points, residuals ~0.1 -> chi^2 = 1.0, dof = 98 -> reduced ~0.01 (good)
    return dict(
        popt=np.array([1.0, 2.0]),
        pcov=np.eye(2),
        residuals=np.full(100, 0.1),
        n_data=100,
        iterations=1000,
        execution_time=0.0,
    )


def test_max_nfev_with_good_chi2_grades_max_iter_via_status() -> None:
    """status==0 (SciPy max_nfev code) + finite reduced chi^2 -> max_iter, not failed."""
    w = _wrapper()
    result = w._create_fit_result(
        convergence_status="failed",
        solver_status=0,
        **_good_chi2_inputs(),
    )
    assert result.convergence_status == "max_iter"
    # quality is graded on the real reduced chi^2 (good here), never forced poor
    assert result.quality_flag in ("good", "marginal")


def test_max_nfev_with_good_chi2_grades_max_iter_via_reason_string() -> None:
    """When no status code is threaded, the SciPy message string also triggers it."""
    w = _wrapper()
    result = w._create_fit_result(
        convergence_status="failed",
        convergence_reason="The maximum number of function evaluations is exceeded.",
        **_good_chi2_inputs(),
    )
    assert result.convergence_status == "max_iter"


def test_genuine_failure_stays_failed() -> None:
    """A non-budget failure (e.g. status=-1, no max_nfev reason) is not upgraded."""
    w = _wrapper()
    result = w._create_fit_result(
        convergence_status="failed",
        solver_status=-1,
        convergence_reason="Improper input parameters.",
        **_good_chi2_inputs(),
    )
    assert result.convergence_status == "failed"


def test_converged_status_is_never_downgraded() -> None:
    """A converged solve stays converged regardless of the threaded reason/status."""
    w = _wrapper()
    result = w._create_fit_result(
        convergence_status="converged",
        solver_status=0,
        convergence_reason="The maximum number of function evaluations is exceeded.",
        **_good_chi2_inputs(),
    )
    assert result.convergence_status == "converged"


def test_grading_is_numerics_safe() -> None:
    """The relabel must not perturb parameters / chi^2 / covariance."""
    w = _wrapper()
    inputs = _good_chi2_inputs()
    result = w._create_fit_result(
        convergence_status="failed",
        solver_status=0,
        **inputs,
    )
    np.testing.assert_array_equal(result.parameters, inputs["popt"])
    assert result.chi_squared == float(np.sum(inputs["residuals"] ** 2))


# --- Phase 6: laminar stratified-LS fourier deletion (Tasks 3 & 4) ---


def _laminar_strat_stub(per_angle_mode: str):
    phi = np.array([0.0, 60.0, 120.0])
    n = 30
    rng = np.random.default_rng(0)
    strat = type(
        "S",
        (),
        {
            "phi_flat": np.repeat(phi, n),
            "t1_flat": np.tile(np.arange(n, dtype=float), 3),
            "t2_flat": np.tile(np.arange(n, dtype=float), 3),
            "g2_flat": 1.0 + 0.3 * rng.random(3 * n),
            "sigma": None,
        },
    )()
    names = ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]
    x0 = np.concatenate([np.full(3, 0.3), np.full(3, 1.0), np.zeros(7)])
    lo = np.concatenate([np.zeros(3), np.full(3, 0.5), np.full(7, -1e6)])
    hi = np.concatenate([np.ones(3), np.full(3, 1.5), np.full(7, 1e6)])
    return strat, names, x0, (lo, hi)


def test_stratified_ls_fourier_config_rejected():
    import logging

    import pytest

    from xpcsjax.optimization.nlsq.strategies.stratified_ls import (
        fit_with_stratified_least_squares,
    )

    strat, names, x0, bounds = _laminar_strat_stub("fourier")
    with pytest.raises(ValueError, match="per_angle_mode"):
        fit_with_stratified_least_squares(
            stratified_data=strat,
            per_angle_scaling=True,
            physical_param_names=names,
            initial_params=x0,
            bounds=bounds,
            log=logging.getLogger("t"),
            anti_degeneracy_config={"enable": True, "per_angle_mode": "fourier"},
            analysis_mode="laminar_flow",
        )


def test_stratified_ls_has_no_fourier_expansion_arm():
    import inspect

    from xpcsjax.optimization.nlsq.strategies import stratified_ls

    src = inspect.getsource(stratified_ls.fit_with_stratified_least_squares)
    assert "use_fourier" not in src
    assert "transform_params_from_fourier" not in src
    assert "get_basis_matrix" not in src
