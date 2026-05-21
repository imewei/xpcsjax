"""Unit tests for _resolve_color_limits."""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.viz.nlsq_plots import _resolve_color_limits


def test_normal_data_returns_percentile_limits() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(loc=1.2, scale=0.05, size=(50, 50))
    vmin, vmax = _resolve_color_limits(data, percentile_min=1.0, percentile_max=99.0)
    assert vmin < vmax
    assert np.isfinite(vmin) and np.isfinite(vmax)


def test_all_nan_returns_fallback() -> None:
    vmin, vmax = _resolve_color_limits(np.full((10, 10), np.nan))
    assert vmin == 1.0 and vmax == 1.5


def test_empty_matrix_returns_fallback() -> None:
    vmin, vmax = _resolve_color_limits(np.zeros((0, 0)))
    assert vmin == 1.0 and vmax == 1.5


def test_flat_constant_matrix_returns_widened_range() -> None:
    vmin, vmax = _resolve_color_limits(np.full((10, 10), 1.25))
    assert vmin == pytest.approx(1.25)
    assert vmax == pytest.approx(vmin + 1.0)


def test_percentile_clamp_excludes_outliers() -> None:
    data = np.ones((10, 10)) * 1.2
    data[0, 0] = 100.0
    data[0, 1] = -100.0
    vmin, vmax = _resolve_color_limits(data, percentile_min=5.0, percentile_max=95.0)
    assert vmin > -50.0 and vmax < 50.0


def test_returns_floats_not_numpy_scalars() -> None:
    vmin, vmax = _resolve_color_limits(np.ones((10, 10)) * 1.2)
    assert isinstance(vmin, float) and isinstance(vmax, float)
