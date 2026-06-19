"""pytest-qt tests for the live-diagnostics widgets."""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from xpcsjax.gui.views.diagnostics_panel import (  # noqa: E402
    BannerList,
    LayerStatusChips,
    SSRCurveWidget,
)


def test_ssr_curve_accumulates_points(qtbot):
    w = SSRCurveWidget()
    qtbot.addWidget(w)
    w.add_point(1, 100.0)
    w.add_point(2, 50.0)
    assert w.point_count() == 2
    w.reset()
    assert w.point_count() == 0


def test_ssr_curve_skips_nonpositive_and_nonfinite(qtbot):
    w = SSRCurveWidget()
    qtbot.addWidget(w)
    w.add_point(1, 10.0)
    w.add_point(2, 0.0)  # log y-axis: 0 has no position -> skipped
    w.add_point(3, float("nan"))  # skipped
    w.add_point(4, -5.0)  # skipped
    assert w.point_count() == 1


def test_layer_chips_reflect_active_map(qtbot):
    w = LayerStatusChips()
    qtbot.addWidget(w)
    w.set_layers({"L1": True, "L2": True, "L3": False, "L4": True, "L5": False})
    assert w.active_layers() == {"L1", "L2", "L4"}


def test_banner_list_accumulates(qtbot):
    w = BannerList()
    qtbot.addWidget(w)
    w.add_banner("ANTI-DEGENERACY: Layer 2 ...", "info")
    w.add_banner("GRADIENT COLLAPSE DETECTED at iteration 7!", "gradient_collapse")
    assert w.count() == 2
    w.clear()
    assert w.count() == 0
