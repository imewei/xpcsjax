"""`_warn_nlsq_bound_saturation` must not misreport NaN uncertainties.

NaN/inf uncertainties are the documented global-escape (CMA-ES / multistart)
sentinel — the covariance solve did not run. They are a real array (not
``None``), and ``float('nan') >= 1e-30`` is ``False``, so the near-zero guard
used to fall through and log a bogus "NLSQ bound saturation: ... +/- 0" line.
"""

import logging

import numpy as np

from xpcsjax.cli.optimization_runner import _warn_nlsq_bound_saturation
from xpcsjax.optimization.nlsq.results import OptimizationResult


def _make_result(uncertainties: np.ndarray) -> OptimizationResult:
    n = uncertainties.size
    return OptimizationResult(
        parameters=np.linspace(1.0, 2.0, n),
        uncertainties=uncertainties,
        covariance=np.full((n, n), np.nan),
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        convergence_status="partial",
        iterations=0,
        execution_time=0.1,
        device_info={},
        quality_flag="marginal",
        nlsq_diagnostics={
            "parameter_names": [f"p{i}" for i in range(n)],
            "global_escape": "cmaes",
        },
    )


def test_nan_uncertainties_do_not_warn_bound_saturation(caplog):
    """A global-escape result (all-NaN uncertainties) emits no saturation warning."""
    result = _make_result(np.full(3, np.nan))
    with caplog.at_level(logging.WARNING, logger="xpcsjax"):
        _warn_nlsq_bound_saturation(result)
    for r in caplog.records:
        msg = r.getMessage()
        assert "bound saturation" not in msg.lower()
        assert "+/- 0" not in msg
        assert "saturated at bounds" not in msg


def test_genuinely_zero_uncertainty_still_warns(caplog):
    """A real near-zero uncertainty must still surface a saturation warning."""
    result = _make_result(np.array([0.0, 1.0, 1.0]))
    with caplog.at_level(logging.WARNING, logger="xpcsjax"):
        _warn_nlsq_bound_saturation(result)
    assert any("bound saturation" in r.getMessage().lower() for r in caplog.records)
