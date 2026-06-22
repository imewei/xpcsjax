"""Square-aspect mixin and heatmap-fitting helpers for plot widgets.

Provides ``_SquareAspectMixin`` (keeps plot widgets square on screen) and
``_fit_square_view`` (configures a heatmap's axes for gap-free square fills).
All GUI-process-side only: pyqtgraph; no JAX.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyqtgraph as pg

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget

    _SquareBase = QWidget
else:
    _SquareBase = object


def _fit_square_view(plot: pg.PlotItem) -> None:
    """Configure a heatmap to fill its tile with t₁ and t₂ on identical ranges.

    Two-time C₂ matrices are square (the same delay axis on both dims), so the
    requirement is that t₁ and t₂ show the *same range*. The fix is therefore the
    opposite of a data-aspect lock: ``setAspectLocked(True)`` keeps square
    *pixels* by inflating one axis's range whenever the tile isn't pixel-perfect
    square (that is what pushed the t₂ axis past the t₁ range). Instead the view
    is left aspect-unlocked and fit to the data on each axis independently — for
    square data that yields identical t₁/t₂ ranges and a gap-free fill, while the
    square widget tile (:class:`_SquareAspectMixin`) keeps the image square.
    Zero default padding removes the ~2 % auto-range margin so the image hugs the
    box edges.
    """
    vb = plot.getViewBox()
    vb.setAspectLocked(False)
    try:
        vb.setDefaultPadding(0.0)
    except Exception:  # noqa: BLE001 - older pyqtgraph builds lack setDefaultPadding
        pass


class _SquareAspectMixin(_SquareBase):
    """Keep a plot widget square on screen by locking height to its width.

    Height-for-width propagation through nested ``QHBoxLayout``/``QVBoxLayout``
    is unreliable in Qt, so squareness is enforced imperatively: every resize
    re-pins the widget's height to its current width. The guard (only act when
    height differs) prevents the set-height → relayout → resize feedback loop.
    Applied to every result plot so the per-φ grid renders uniform square tiles.
    """

    def resizeEvent(self, ev) -> None:  # noqa: N802, ANN001 - Qt event override
        super().resizeEvent(ev)
        side = self.width()
        if side > 0 and self.height() != side:
            self.setFixedHeight(side)
