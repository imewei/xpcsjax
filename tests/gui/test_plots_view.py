"""Tests for interactive PyQtGraph plot widgets (plots_view)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from xpcsjax.gui.views.plots_view import (  # noqa: E402
    PerAngleOverlayView,
    ResidualMapView,
    ResultPlots,
    TwoTimeMapView,
)
from xpcsjax.gui.viz_bundle import VizBundle  # noqa: E402, I001

# ---------------------------------------------------------------------------
# Verbatim tests from brief
# ---------------------------------------------------------------------------


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


def test_result_plots_sets_bundle(qtbot):
    w = ResultPlots()
    qtbot.addWidget(w)
    exp = np.random.default_rng(0).random((2, 16, 16))
    w.set_bundle(
        VizBundle(
            exp_c2=exp,
            residuals=np.zeros_like(exp),
            phi_angles=np.array([0.0, 45.0]),
        )
    )
    assert w.phi_count() == 2
    assert w.two_time().has_image()


# ---------------------------------------------------------------------------
# Graceful-degradation tests (required by brief)
# ---------------------------------------------------------------------------


def test_set_bundle_none_does_not_crash(qtbot):
    """set_bundle(None) must not raise; phi_count() == 0."""
    w = ResultPlots()
    qtbot.addWidget(w)
    w.set_bundle(None)
    assert w.phi_count() == 0


def test_bundle_missing_optional_fields_does_not_crash(qtbot):
    """A bundle with residuals=None, model_c2=None, phi_angles=None must render without error."""
    w = ResultPlots()
    qtbot.addWidget(w)
    exp = np.random.default_rng(1).random((3, 8, 8))
    # All optional fields absent
    bundle = VizBundle(exp_c2=exp)
    w.set_bundle(bundle)
    # phi_count derived from exp_c2 leading dim when phi_angles is None
    assert w.phi_count() == 3
    # two-time view must still render the selected phi slice
    assert w.two_time().has_image()
