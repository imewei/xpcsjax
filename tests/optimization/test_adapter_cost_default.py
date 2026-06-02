"""Adapter must not mint a 'good' fit from a missing objective.

Quality-gate finding: ``info.get("cost", 0.0)`` defaulted chi-squared to 0.0
when neither ``"fun"`` nor ``"cost"`` was present, so a solve that returned no
objective produced ``reduced_chi_squared == 0`` and the best possible quality
flag. A missing objective must yield a non-finite chi-squared, never zero.
"""

from __future__ import annotations

import numpy as np

from xpcsjax.optimization.nlsq.adapter import NLSQAdapter


def _result(info):
    adapter = NLSQAdapter()
    return adapter._convert_nlsq_result(
        popt=np.array([1.0, 2.0]),
        pcov=np.eye(2),
        info=info,
        n_data=100,
        execution_time=0.01,
    )


def test_missing_objective_yields_nonfinite_chi2_not_good():
    res = _result({"success": True})  # no "fun", no "cost"
    assert not np.isfinite(res.chi_squared)
    assert res.quality_flag != "good"


def test_explicit_cost_still_used():
    res = _result({"success": True, "cost": 0.5 * 50.0})  # cost = 0.5 * SSR
    assert np.isclose(res.chi_squared, 50.0)
