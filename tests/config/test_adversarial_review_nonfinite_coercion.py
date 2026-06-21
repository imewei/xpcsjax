"""Regression tests for the 2026-06-20 adversarial-review (codex) findings.

Finding #1 (P1) — non-finite config numerics bypassed the validator gate. The
config-debug pass added ``_is_non_finite`` rejection to the *physics validators*,
but :meth:`ConfigManager._validate_config` is logging-only and never invokes
them, so the value-extraction methods that coerce with a bare ``float(...)``
(``get_initial_parameters`` initial values + per-angle scaling, and
``ParameterManager`` bounds) still emitted ``NaN`` / ``±inf`` straight into the
optimizer x0 / bounds. These must raise at the coercion boundary instead.

Finding #2 (P2) — the first-lag quality check sampled a single cell
``matrix[0, 1]`` rather than the first-superdiagonal population, so one boundary
artifact in an otherwise-clean angle could drag the score below a strict
threshold and silently drop valid data. The check must use a robust aggregate
(finite median) over the whole first superdiagonal.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xpcsjax.config.manager import ConfigManager
from xpcsjax.config.parameter_manager import ParameterManager
from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.data.filtering_utils import XPCSDataFilter


# --------------------------------------------------------------------------- #
# Finding #1 — non-finite numerics must be rejected at coercion, not returned
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_initial_parameter_value_rejects_nonfinite(bad: float) -> None:
    cfg = ConfigManager(
        config_override={
            "analysis_mode": "static_anisotropic",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [bad, 0.5, 10.0],
            },
        }
    )
    with pytest.raises(ValueError, match="non-finite"):
        cfg.get_initial_parameters()


@pytest.mark.parametrize("key", ["contrast", "offset"])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_per_angle_scaling_rejects_nonfinite(key: str, bad: float) -> None:
    cfg = ConfigManager(
        config_override={
            "analysis_mode": "laminar_flow",
            "initial_parameters": {
                "parameter_names": ["D0"],
                "values": [1000.0],
                "per_angle_scaling": {key: [1.0, bad, 1.0]},
            },
        }
    )
    with pytest.raises(ValueError, match="non-finite"):
        cfg.get_initial_parameters()


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("bound_key", ["min", "max"])
def test_bounds_reject_nonfinite(bad: float, bound_key: str) -> None:
    config = {
        "analysis_mode": "static_anisotropic",
        "parameter_space": {"bounds": [{"name": "D0", bound_key: bad}]},
    }
    with pytest.raises(ValueError, match="non-finite"):
        ParameterManager(config, AnalysisMode.STATIC_ANISOTROPIC)


def test_finite_initial_parameters_still_load() -> None:
    # Guard against over-rejection: a healthy config still resolves cleanly.
    cfg = ConfigManager(
        config_override={
            "analysis_mode": "static_anisotropic",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000.0, 0.5, 10.0],
            },
        }
    )
    params = cfg.get_initial_parameters()
    assert params["D0"] == pytest.approx(1000.0)
    assert params["alpha"] == pytest.approx(0.5)


def test_finite_bounds_still_load() -> None:
    config = {
        "analysis_mode": "static_anisotropic",
        "parameter_space": {"bounds": [{"name": "D0", "min": 1.0, "max": 1.0e5}]},
    }
    pm = ParameterManager(config, AnalysisMode.STATIC_ANISOTROPIC)
    assert pm._default_bounds["D0"]["min"] == pytest.approx(1.0)
    assert pm._default_bounds["D0"]["max"] == pytest.approx(1.0e5)


# --------------------------------------------------------------------------- #
# Finding #2 — first-lag quality uses the superdiagonal population, not one cell
# --------------------------------------------------------------------------- #
def test_quality_score_robust_to_single_boundary_artifact() -> None:
    """A clean angle with ONE bad first-lag cell must not lose diagonal quality.

    Pre-fix the score read ``matrix[0, 1]`` alone; a single zeroed boundary pair
    (``matrix[0,1]=0``) dragged ``diagonal_quality`` from 1.0 to 0.5 even though
    the lag-1 population is overwhelmingly valid (~1.3).
    """
    n = 8
    m = np.full((n, n), 1.3, dtype=np.float64)
    np.fill_diagonal(m, 2.4)  # tau=0 self-correlation spike (excluded)
    m = (m + m.T) / 2.0
    m[0, 1] = m[1, 0] = 0.0  # single boundary artifact in the first superdiagonal

    f = XPCSDataFilter()
    score = f._calculate_matrix_quality_score(m)
    # Median of the first superdiagonal is ~1.3 (in [0.5, 2.0]) → diagonal_quality
    # stays 1.0. With finite_fraction=1.0, symmetry≈1.0, range_quality=1.0 the
    # overall score is ~1.0; the pre-fix single-cell read produced ~0.85.
    assert score > 0.95


def test_quality_score_still_flags_genuinely_overnormalized() -> None:
    # The whole first-lag population sits above the Siegert ceiling → penalized.
    n = 8
    m = np.full((n, n), 3.5, dtype=np.float64)
    np.fill_diagonal(m, 3.5)
    f = XPCSDataFilter()
    score = f._calculate_matrix_quality_score(m)
    assert score < 1.0
