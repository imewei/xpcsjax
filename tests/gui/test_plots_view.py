"""Tests for interactive PyQtGraph plot widgets (plots_view)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from xpcsjax.gui.views.plots_view import (  # noqa: E402
    PerAngleOverlayView,
    ResidualMapView,
    TwoTimeMapView,
)

# ---------------------------------------------------------------------------
# Sub-widget tests (the reusable primitives PhiResultsGrid composes)
# ---------------------------------------------------------------------------


def test_superdiag_mean_handles_degenerate_matrix():
    # The per-angle overlay scalar is the mean of the first superdiagonal (tau=dt).
    # A 2x2 has a one-element superdiagonal; a 1x1 (or empty) has none. The latter
    # must return NaN WITHOUT a RuntimeWarning (np.mean of an empty slice warns).
    import warnings

    from xpcsjax.gui.views.plots_view import _superdiag_mean

    assert _superdiag_mean(np.array([[1.0, 2.0], [3.0, 4.0]])) == 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning becomes a hard failure
        val = _superdiag_mean(np.array([[5.0]]))
    assert np.isnan(val)


def test_two_time_map_accepts_array(qtbot):
    w = TwoTimeMapView()
    qtbot.addWidget(w)
    w.show_map(np.random.default_rng(0).random((50, 50)))
    assert w.has_image()


def test_residual_map_accepts_array(qtbot):
    w = ResidualMapView()
    qtbot.addWidget(w)
    w.show_map(np.zeros((20, 20)))
    assert w.has_image()


def test_per_angle_overlay_plots_curves(qtbot):
    w = PerAngleOverlayView()
    qtbot.addWidget(w)
    w.show_overlay(
        np.array([0.0, 45.0]),
        exp_g2=np.array([1.3, 1.2]),
        model_g2=np.array([1.29, 1.21]),
    )
    assert w.curve_count() == 2

# (ResultPlots removed in the redesign — per-phi grid behavior is covered by
# tests/gui/test_gui_redesign.py::PhiResultsGrid tests.)
