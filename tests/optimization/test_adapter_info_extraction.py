"""Regression: NLSQAdapter.fit() must read real diagnostics off CurveFitResult.

Bug: `fit()` did `info = getattr(result, "info", {})`, assuming diagnostics
nest under `.info`. `CurveFitResult` (installed `nlsq` package) is itself
dict-like with success/cost/fun/nfev/status/message as TOP-LEVEL keys, so
`.info` never exists and `info` was always `{}` -- every real fit reported
convergence_status="failed", chi_squared=nan, quality_flag="poor" regardless
of the true outcome. Fix: build `info` from `dict(result)`.

Also covers the sibling bug in `_convert_nlsq_result`: the chi-squared
fallback used `isinstance(raw_fun, np.ndarray)`, which a JAX array fails,
silently switching to `chi_squared = 2 * cost` -- only exact for linear
loss, but the adapter defaults to loss="soft_l1" (robust), where
cost = 0.5*sum(rho(r^2)) != 0.5*sum(r^2).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from nlsq.result.curve_fit_result import CurveFitResult

from xpcsjax.optimization.nlsq.adapter import NLSQAdapter

PHI = np.array([0.0, 45.0, 90.0])
N_T = 4
T1 = np.tile(np.arange(N_T, dtype=float), len(PHI))
T2 = T1.copy()
G2 = np.ones_like(T1)
DATA = {"phi": PHI, "t1": T1, "t2": T2, "g2": G2}


def _make_adapter_with_stubbed_curve_fit(curve_fit_result):
    adapter = NLSQAdapter()
    # Stub out the physics model build (heavy XPCS-model setup, irrelevant to
    # the info-extraction bug under test) and the NLSQ solve itself, so the
    # real fit() code path -- including the result-unpacking block -- runs
    # against a controlled CurveFitResult.
    adapter._build_model_function = lambda **kwargs: (  # type: ignore[method-assign]
        lambda x, *params: np.ones(len(x)),
        False,
        False,
    )
    adapter._fitter.curve_fit = lambda **kwargs: curve_fit_result  # type: ignore[method-assign]
    return adapter


def test_fit_reads_top_level_diagnostics_not_empty_info():
    """Fix 1: a converged CurveFitResult must not be reported as failed."""
    popt = np.array([1.0, 2.0])
    fun = jnp.array([0.1, -0.2, 0.05, 0.0, 0.1, -0.1])
    result = CurveFitResult(
        x=popt,
        success=True,
        cost=float(0.5 * np.sum(np.asarray(fun) ** 2)),
        fun=fun,
        nfev=12,
        status=1,
        message="Converged",
        pcov=np.eye(len(popt)),
    )
    adapter = _make_adapter_with_stubbed_curve_fit(result)
    opt_result = adapter.fit(
        data=DATA,
        config={},
        initial_params=popt,
        bounds=(np.array([0.0, 0.0]), np.array([10.0, 10.0])),
    )
    assert opt_result.convergence_status != "failed"
    assert np.isfinite(opt_result.chi_squared)
    assert not np.isnan(opt_result.chi_squared)


def test_convert_nlsq_result_jax_array_fun_uses_sum_of_squares_not_cost_times_two():
    """Fix 2: JAX-array `fun` under soft_l1 must not fall back to 2*cost."""
    adapter = NLSQAdapter()
    fun = jnp.array([1.0, 2.0, 3.0])
    sum_sq = float(np.sum(np.asarray(fun) ** 2))
    # Robust-loss cost is NOT 0.5*sum(fun**2) -- pick a value that would
    # diverge sharply from sum_sq if the buggy 2*cost path were taken.
    robust_cost = 0.5 * sum_sq * 0.3
    info = {"success": True, "cost": robust_cost, "fun": fun, "nfev": 5, "status": 1}
    res = adapter._convert_nlsq_result(
        popt=np.array([1.0, 2.0]),
        pcov=np.eye(2),
        info=info,
        n_data=100,
        execution_time=0.01,
    )
    assert np.isclose(res.chi_squared, sum_sq)
    assert not np.isclose(res.chi_squared, 2 * robust_cost)
