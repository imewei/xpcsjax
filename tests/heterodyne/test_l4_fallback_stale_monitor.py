"""Regression: L4 must not report the discarded adapter monitor on fallback.

Bug (pre-fix): in ``heterodyne_core.py`` the two joint-fit paths build an L4
gradient-collapse monitor + callback once and pass the callback ONLY to the
``NLSQAdapter``. When the adapter fires that callback at least once (recording a
per-iteration observation in the monitor) and then returns ``success=False``,
the surrounding code raises and falls back to the ``NLSQWrapper``, which runs
WITHOUT any callback. The wrapper's parameters become the returned result, but
``_assemble_l4_extras`` still trusts the monitor — so the returned
``gradient_monitor`` block reports ``mechanism="per_iteration_gradient_ratio"``
describing a run whose parameters were thrown away.

The fix forces the post-solve covariance-condition block (computed from the
ACTUAL returned ``joint_result``) whenever the returned result did NOT come from
the monitored adapter, so ``mechanism`` honestly reports ``post_solve_fallback``.

This is diagnostics-only: the fit (params / chi^2) is the wrapper's and is
unchanged by the fix.
"""

from __future__ import annotations

import numpy as np

import xpcsjax.optimization.nlsq.heterodyne_core as hc
from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi
from xpcsjax.optimization.nlsq.heterodyne_results import NLSQResult


def _make_failing_adapter(orig_adapter_cls):
    """Build an NLSQAdapter subclass whose ``fit`` fires the L4 callback once
    then returns ``success=False`` — forcing the unmonitored wrapper fallback.
    """

    class _FailingAdapter(orig_adapter_cls):  # type: ignore[misc, valid-type]
        def fit(  # type: ignore[override]
            self,
            residual_fn,
            initial_params,
            bounds,
            config,
            jacobian_fn=None,
            callback=None,
        ):
            p = np.asarray(initial_params, dtype=np.float64)
            if callback is not None:
                # Fire the per-iteration callback once so the monitor records
                # >= 1 observation (the precondition for the stale-monitor bug).
                cost = float(np.sum(np.asarray(residual_fn(p), dtype=np.float64) ** 2))
                callback(0, cost, p, None)
            return NLSQResult(
                parameters=p,
                parameter_names=list(self._parameter_names),
                success=False,
                message="forced failure (regression fixture)",
                convergence_reason="failed",
            )

    return _FailingAdapter


def test_fallback_does_not_report_discarded_adapter_monitor(monkeypatch):
    """Adapter fires the L4 callback then fails; the wrapper fallback succeeds.

    The returned ``gradient_monitor`` block must report ``post_solve_fallback``
    (honest, computed from the wrapper's result), NOT the discarded adapter
    run's ``per_iteration_gradient_ratio``.
    """
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "averaged",
            "enable_gradient_monitoring": True,
        }
    )

    monkeypatch.setattr(hc, "NLSQAdapter", _make_failing_adapter(hc.NLSQAdapter))

    res = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)
    gm = res.nlsq_diagnostics["gradient_monitor"]

    assert gm["mechanism"] == "post_solve_fallback", (
        "fallback path must not surface the discarded adapter monitor's "
        f"per-iteration ratios; got mechanism={gm['mechanism']!r}"
    )
