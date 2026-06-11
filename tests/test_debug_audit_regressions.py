"""Regression tests for the 2026-06-10 whole-codebase debug audit.

Each test pins a specific confirmed-and-fixed defect so it cannot silently
regress. Tests are deliberately lightweight (pure functions / small inputs) and
reference the finding they guard in the docstring.
"""

from __future__ import annotations

import numpy as np


def test_get_group_indices_scaling_resolves() -> None:
    """Audit [26]: get_group_indices('scaling') must resolve, not KeyError.

    The 'scaling' group (contrast, offset) lives in the 16-element with-scaling
    array at indices 14, 15; physics-group indices are unchanged.
    """
    from xpcsjax.config.heterodyne_parameter_names import get_group_indices

    assert get_group_indices("scaling") == (14, 15)
    # Physics groups keep their canonical 0-based positions.
    assert get_group_indices("reference") == (0, 1, 2)
    assert get_group_indices("angle") == (13,)


def test_combine_angle_results_excludes_nonfinite_covariance() -> None:
    """Audit [5]: a non-finite per-angle covariance must not poison the combined
    inverse-variance covariance (it is already zero-weighted in the params)."""
    from xpcsjax.optimization.nlsq.strategies.sequential import combine_angle_results

    per_angle = [
        {
            "success": True,
            "parameters": np.array([1.0, 2.0]),
            "covariance": np.diag([0.1, 0.2]),
            "n_points": 100,
            "cost": 1.0,
        },
        {
            "success": True,
            "parameters": np.array([1.1, 2.1]),
            "covariance": np.diag([np.nan, np.inf]),  # failed solve
            "n_points": 100,
            "cost": 1.0,
        },
    ]

    params, cov, _cost = combine_angle_results(per_angle, weighting="inverse_variance")

    assert np.all(np.isfinite(params)), "params poisoned by non-finite angle"
    assert np.all(np.isfinite(cov)), "combined covariance poisoned by non-finite angle"


def test_parameter_manager_two_component_default_params() -> None:
    """Audit [8]: the active-parameter fallback for two_component must return the
    heterodyne parameter set, not the laminar_flow set."""
    from xpcsjax.config.parameter_manager import ParameterManager

    pm = ParameterManager({}, "two_component")
    params = pm._get_default_active_parameters()

    assert "D0_ref" in params
    assert "phi0_het" in params
    # Must NOT fall through to the laminar_flow parameter list.
    assert "gamma_dot_t0" not in params


def test_absent_analysis_mode_resolves_to_isotropic_consistently() -> None:
    """Audit [25]: with analysis_mode absent, the .analysis_mode property and the
    cached ParameterManager must agree (canonical default: static_isotropic)."""
    from xpcsjax.config.manager import ConfigManager
    from xpcsjax.config.parameter_registry import AnalysisMode

    cm = ConfigManager(config_override={"analyzer_parameters": {}})  # no analysis_mode key

    assert cm.analysis_mode == AnalysisMode.STATIC_ISOTROPIC
    pm_mode = str(cm._get_parameter_manager().analysis_mode).lower()
    assert "isotropic" in pm_mode
    assert "anisotropic" not in pm_mode

