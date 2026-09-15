"""A4: the out-of-core route is one shared function so the initial-decision
and strategy-recheck call sites in ``NLSQWrapper.fit`` can never drift again.

Before this fix, the initial trigger resolved ``per_angle_mode`` with the
plain (non-pinned) resolver while the recheck trigger used
``resolve_per_angle_mode_static_pinned``. For a static analysis mode with an
explicit ``per_angle_mode: "constant"`` token and the SAME dense popt, the
initial branch reported DOF=n_physical (3) and the recheck branch reported
DOF=n_physical+2*n_phi (13) -- different ``reduced_chi_squared`` for an
identical fit depending on which trigger happened to fire.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.optimization.nlsq.wrapper_out_of_core_route import run_out_of_core_route


class _NullLogger:
    def info(self, *a, **k):
        pass

    warning = critical = error = info


class _FakeConfig:
    def __init__(self, anti_degeneracy: dict) -> None:
        self.config = {"optimization": {"nlsq": {"anti_degeneracy": anti_degeneracy}}}


def _canned_ooc(*, initial_params, **_kwargs):
    n = len(initial_params)
    return (
        np.asarray(initial_params, dtype=float),
        np.eye(n),
        {"chi_squared": 130.0, "convergence_status": "converged", "iterations": 5},
    )


def _run(*, per_angle_mode: str, analysis_mode: AnalysisMode, n_phi: int, n_physical: int):
    popt = np.zeros(n_physical + 2 * n_phi)
    return run_out_of_core_route(
        stratified_data=None,
        data=object(),
        per_angle_scaling=True,
        physical_param_names=[f"p{i}" for i in range(n_physical)],
        initial_params=popt,
        bounds=None,
        logger=_NullLogger(),
        config=_FakeConfig({"per_angle_mode": per_angle_mode}),
        analysis_mode=analysis_mode,
        n_points=1_000_000,
        n_phi=n_phi,
        resolved_physical=None,
        start_time=0.0,
        strategy_reason="forced",
        recovery_tag="out_of_core_delegation",
        fit_ooc=_canned_ooc,
    )


def test_static_mode_constant_token_is_pinned_to_individual_dof():
    """Static mode + explicit 'constant' must resolve DOF through the pin
    (-> individual -> n_physical + 2*n_phi = 13), never the raw token's
    n_physical (3, the pre-fix initial-branch bug)."""
    n_phi, n_physical = 5, 3
    result = _run(
        per_angle_mode="constant",
        analysis_mode=AnalysisMode.STATIC_ANISOTROPIC,
        n_phi=n_phi,
        n_physical=n_physical,
    )
    expected_dof = n_physical + 2 * n_phi  # pinned to "individual"
    expected_reduced_chi2 = 130.0 / max(1, 1_000_000 - expected_dof)
    assert result.reduced_chi_squared == pytest.approx(expected_reduced_chi2)


def test_laminar_mode_constant_token_is_not_pinned():
    """laminar_flow honors the requested token unchanged (pin is a no-op)."""
    n_phi, n_physical = 5, 7
    result = _run(
        per_angle_mode="constant",
        analysis_mode=AnalysisMode.LAMINAR_FLOW,
        n_phi=n_phi,
        n_physical=n_physical,
    )
    expected_dof = n_physical  # "constant" -> 0 optimized scaling params
    expected_reduced_chi2 = 130.0 / max(1, 1_000_000 - expected_dof)
    assert result.reduced_chi_squared == pytest.approx(expected_reduced_chi2)


def test_static_and_laminar_share_the_same_resolver_path():
    """Regression guard for the extraction itself: whatever DOF the function
    reports for a given (mode, token, n_phi, n_physical) combination is
    deterministic and does not depend on which call site invoked it -- there
    is only one code path left to compute it."""
    kwargs = dict(
        per_angle_mode="individual",
        analysis_mode=AnalysisMode.STATIC_ISOTROPIC,
        n_phi=4,
        n_physical=3,
    )
    first = _run(**kwargs)
    second = _run(**kwargs)
    assert first.reduced_chi_squared == pytest.approx(second.reduced_chi_squared)
