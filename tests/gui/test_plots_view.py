"""Tests for interactive PyQtGraph plot widgets (plots_view)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

from xpcsjax.gui.views.plots_view import (  # noqa: E402
    _SCATTER_MAX_POINTS,
    DiagonalResidualView,
    PhiResultsGrid,
    ResidualHistogramView,
    ResidualMapView,
    ResidualsVsFittedView,
    TwoTimeMapView,
    _c2_levels,
    _residual_levels,
    _time_rect,
)

# ---------------------------------------------------------------------------
# Sub-widget tests (the reusable primitives PhiResultsGrid composes)
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


@pytest.mark.parametrize("view_cls", [TwoTimeMapView, ResidualMapView])
def test_map_view_survives_unresolvable_colormap(qtbot, monkeypatch, view_cls):
    """Colorbar construction must degrade gracefully, like the image item's own
    colormap resolution, when pg.colormap.get() can't resolve a name — not
    crash widget construction (regression: the colorbar's resolver call was
    unguarded, unlike the sibling _apply_colormap call it sits next to)."""

    def _raise(*_a, **_k):
        raise RuntimeError("colormap unavailable in this environment")

    monkeypatch.setattr(pg.colormap, "get", _raise)
    w = view_cls()
    qtbot.addWidget(w)
    w.show_map(np.random.default_rng(0).random((30, 30)))
    assert w.has_image()


# ---------------------------------------------------------------------------
# Colormap + t₁/t₂ axes (the changes requested over the grayscale frame-index view)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("view_cls", [TwoTimeMapView, ResidualMapView])
def test_map_axes_labelled_t1_t2(qtbot, view_cls):
    w = view_cls()
    qtbot.addWidget(w)
    assert "t₁" in w._plot.getAxis("bottom").labelText  # x = t₁
    assert "t₂" in w._plot.getAxis("left").labelText  # y = t₂


@pytest.mark.parametrize("view_cls", [TwoTimeMapView, ResidualMapView])
def test_map_applies_color_lookup_table(qtbot, view_cls):
    # A colormap (jet / RdBu_r) is installed on the ImageItem — no longer the
    # default grayscale. getColorMap() returns the active pg.ColorMap.
    w = view_cls()
    qtbot.addWidget(w)
    assert w._image_item.getColorMap() is not None


def test_two_time_map_places_image_on_physical_t1_t2_grid(qtbot):
    w = TwoTimeMapView()
    qtbot.addWidget(w)
    t1 = np.linspace(0.0, 2.5, 32)  # horizontal extent (width), x = t₁
    t2 = np.linspace(0.0, 5.0, 32)  # vertical extent (height), y = t₂
    w.show_map(np.full((32, 32), 1.2), t1=t1, t2=t2)
    mapped = w._image_item.mapRectToParent(w._image_item.boundingRect())
    assert mapped.width() == pytest.approx(2.5)
    assert mapped.height() == pytest.approx(5.0)


def test_two_time_map_falls_back_to_frame_indices(qtbot):
    # No usable time axes → image keeps its pixel-index extent (still labelled t₁/t₂).
    w = TwoTimeMapView()
    qtbot.addWidget(w)
    w.show_map(np.full((40, 40), 1.2))
    mapped = w._image_item.mapRectToParent(w._image_item.boundingRect())
    assert mapped.width() == pytest.approx(40.0)
    assert mapped.height() == pytest.approx(40.0)


@pytest.mark.parametrize(
    "view_cls",
    [
        TwoTimeMapView,
        ResidualMapView,
        ResidualHistogramView,
        DiagonalResidualView,
        ResidualsVsFittedView,
    ],
)
def test_plot_widgets_render_square(qtbot, view_cls):
    # Every result plot locks height to width on resize, so the per-φ grid tiles
    # render square regardless of the (wider, shorter) space the layout hands them.
    w = view_cls()
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    w.resize(300, 120)  # wide + short -> resizeEvent must re-pin height to width
    qtbot.waitUntil(lambda: w.height() == w.width(), timeout=2000)
    assert w.height() == w.width()


@pytest.mark.parametrize("view_cls", [TwoTimeMapView, ResidualMapView])
def test_map_keeps_equal_t1_t2_range_without_aspect_lock(qtbot, view_cls):
    # Two-time C₂ is square (same delay axis on both dims), so t₁ and t₂ must show
    # the SAME range. Aspect-locking would inflate one axis's range to keep square
    # pixels in a non-square tile (the t₂→150 bug); fitting each axis to the data
    # keeps the ranges identical and fills the tile gap-free.
    w = view_cls()
    qtbot.addWidget(w)
    vb = w._plot.getViewBox()
    assert not vb.state["aspectLocked"]  # no data-aspect coupling
    assert vb.state["defaultPadding"] == pytest.approx(0.0)  # no letterbox padding
    w.show_map(np.full((64, 64), 1.2))  # square matrix -> identical t₁/t₂ spans
    (x0, x1), (y0, y1) = vb.viewRange()
    assert (x1 - x0) == pytest.approx(y1 - y0)


def test_phi_grid_pins_scrollbars_to_keep_square_tiles(qtbot):
    # The vertical scrollbar is pinned on (and the horizontal off) so the
    # viewport width never toggles — otherwise the per-tile height:=width squaring
    # oscillates and tiles render rectangular mid-resize.
    grid = PhiResultsGrid()
    qtbot.addWidget(grid)
    assert grid._scroll.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOn
    assert grid._scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff


def test_c2_levels_clamp_to_unit_band():
    # The bright τ=0 diagonal spike (~2.4) must not blow out the [1.0, 1.5] window.
    arr = np.full((8, 8), 1.0)
    arr[np.diag_indices(8)] = 2.4
    assert _c2_levels(arr) == (1.0, 1.5)


def test_residual_levels_are_symmetric_about_zero():
    lo, hi = _residual_levels(np.array([[-0.3, 0.1], [0.05, 0.2]]))
    assert lo == pytest.approx(-hi)
    assert hi > 0.0


def test_time_rect_none_on_degenerate_axes():
    assert _time_rect(None, np.arange(4.0)) is None
    assert _time_rect(np.array([1.0]), np.array([1.0])) is None  # too short
    assert _time_rect(np.array([np.nan, 1.0]), np.array([0.0, 1.0])) is None  # non-finite


# ---------------------------------------------------------------------------
# Interactive residual diagnostics (replaced the static residuals PNG)
# ---------------------------------------------------------------------------


def test_residual_histogram_renders_and_tolerates_all_nan(qtbot):
    w = ResidualHistogramView()
    qtbot.addWidget(w)
    assert "Residual Value" in w.getAxis("bottom").labelText
    w.show_distribution(np.random.default_rng(0).normal(0.0, 0.05, (30, 30)))
    assert len(w.plotItem.listDataItems()) >= 1  # histogram (+ normal overlay)
    w.show_distribution(np.full((4, 4), np.nan))  # no finite values -> no curves, no raise
    assert w.plotItem.listDataItems() == []


def test_diagonal_residual_traces_diag_against_t1(qtbot):
    w = DiagonalResidualView()
    qtbot.addWidget(w)
    assert "t₁" in w.getAxis("bottom").labelText
    res = np.random.default_rng(0).normal(0.0, 0.05, (16, 16))
    w.show_diagonal(res, np.linspace(0.0, 1.5, 16))
    curves = w.plotItem.listDataItems()
    assert len(curves) == 1
    x = curves[0].getData()[0]
    assert len(x) == 16  # one point per diagonal lag
    assert x[-1] == pytest.approx(1.5)  # physical t₁ axis, not frame index


def test_residuals_vs_fitted_decimates_large_cloud(qtbot):
    w = ResidualsVsFittedView()
    qtbot.addWidget(w)
    n = _SCATTER_MAX_POINTS + 5000
    fitted = np.random.default_rng(0).random((n, 1))
    residuals = np.random.default_rng(1).normal(0.0, 0.05, (n, 1))
    w.show_scatter(fitted, residuals)
    scatters = [it for it in w.plotItem.items if isinstance(it, pg.ScatterPlotItem)]
    assert len(scatters) == 1
    assert len(scatters[0].data) == _SCATTER_MAX_POINTS  # display-only decimation cap


def test_residuals_vs_fitted_tolerates_all_nan(qtbot):
    w = ResidualsVsFittedView()
    qtbot.addWidget(w)
    w.show_scatter(np.full((4, 4), np.nan), np.full((4, 4), np.nan))  # no finite pairs, no raise
    scatters = [it for it in w.plotItem.items if isinstance(it, pg.ScatterPlotItem)]
    assert scatters == []


# (ResultPlots removed in the redesign — per-phi grid behavior is covered by
# tests/gui/test_gui_redesign.py::PhiResultsGrid tests.)
