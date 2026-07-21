"""Two-time map and residual map view widgets.

``TwoTimeMapView`` and ``ResidualMapView`` — pan/zoom-able heatmap views for
two-time correlation matrices and residual surfaces. All GUI-process-side only:
numpy + Qt + pyqtgraph; no JAX.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget

from xpcsjax.gui.views.raster import rasterize

from .helpers import (
    _C2_COLORMAP,
    _RESIDUAL_COLORMAP,
    _apply_colormap,
    _c2_levels,
    _residual_levels,
    _resolve_colormap,
    _time_rect,
)
from .squares import _fit_square_view, _SquareAspectMixin


class TwoTimeMapView(_SquareAspectMixin, pg.GraphicsLayoutWidget):
    """Display a two-time correlation matrix as a pan/zoom-able image.

    Parameters
    ----------
    parent : QWidget or None
        Optional parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._plot = self.addPlot()
        # Fit each axis to the data (no aspect lock) so t₁ and t₂ keep identical
        # ranges and the image fills the square widget tile gap-free.
        _fit_square_view(self._plot)
        # Col-major so array axis 0 (t₁) is horizontal and axis 1 (t₂) vertical.
        self._image_item = pg.ImageItem()
        self._image_item.setOpts(axisOrder="col-major")
        _apply_colormap(self._image_item, _C2_COLORMAP)
        self._plot.addItem(self._image_item)
        self._plot.setLabel("bottom", "t₁")
        self._plot.setLabel("left", "t₂")
        # Levels are auto-scaled per tile (see show_map / _c2_levels), so without
        # a colorbar the same color can mean different values on two different
        # angle tiles with no way to tell — color would be the sole, unreadable
        # carrier of the actual correlation value.
        self._colorbar = pg.ColorBarItem(colorMap=_resolve_colormap(_C2_COLORMAP))
        self._colorbar.setImageItem(self._image_item, insert_in=self._plot)
        self._has_image = False

    def show_map(
        self,
        c2_2d: np.ndarray,
        t1: np.ndarray | None = None,
        t2: np.ndarray | None = None,
    ) -> None:
        """Rasterize and display a 2-D correlation matrix.

        Parameters
        ----------
        c2_2d : np.ndarray
            Two-dimensional array (n_t1, n_t2) of correlation values.
        t1, t2 : np.ndarray or None
            Time axes (seconds). When both are usable the image is placed on a
            physical t₁/t₂ grid; otherwise the axes fall back to frame indices
            (still labelled t₁/t₂).
        """
        # Compute the color window from the FULL-resolution surface, then
        # rasterize for display only — block-mean decimation would otherwise
        # shrink the window so the map disagrees with the full-res diagnostics
        # (histogram/diagonal/scatter) shown alongside it.
        full = np.asarray(c2_2d, dtype=float)
        arr = rasterize(full)
        levels = _c2_levels(full)
        self._image_item.setImage(arr, autoLevels=False)
        self._image_item.setLevels(levels)
        self._colorbar.setLevels(low=levels[0], high=levels[1])
        rect = _time_rect(t1, t2)
        if rect is not None:
            self._image_item.setRect(rect)
        # Fit the view to the image extent exactly (padding=0): t₁/t₂ ranges follow
        # the data, so a square C₂ matrix gets identical axis ranges and a full fill.
        self._plot.getViewBox().autoRange(padding=0.0)
        self._has_image = True

    def clear_map(self) -> None:
        """Remove any displayed image (so a stale map is never shown for new input)."""
        self._image_item.clear()
        self._has_image = False

    def has_image(self) -> bool:
        """Return ``True`` after at least one successful ``show_map`` call."""
        return self._has_image


class ResidualMapView(_SquareAspectMixin, pg.GraphicsLayoutWidget):
    """Display a residual map as a pan/zoom-able image.

    Parameters
    ----------
    parent : QWidget or None
        Optional parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._plot = self.addPlot()
        # Fit each axis to the data (no aspect lock) so t₁ and t₂ keep identical
        # ranges and the image fills the square widget tile gap-free.
        _fit_square_view(self._plot)
        # Col-major so array axis 0 (t₁) is horizontal and axis 1 (t₂) vertical.
        self._image_item = pg.ImageItem()
        self._image_item.setOpts(axisOrder="col-major")
        _apply_colormap(self._image_item, _RESIDUAL_COLORMAP)
        self._plot.addItem(self._image_item)
        self._plot.setLabel("bottom", "t₁")
        self._plot.setLabel("left", "t₂")
        self._colorbar = pg.ColorBarItem(colorMap=_resolve_colormap(_RESIDUAL_COLORMAP))
        self._colorbar.setImageItem(self._image_item, insert_in=self._plot)
        self._has_image = False

    def show_map(
        self,
        residual_2d: np.ndarray,
        t1: np.ndarray | None = None,
        t2: np.ndarray | None = None,
    ) -> None:
        """Rasterize and display a 2-D residual matrix.

        Parameters
        ----------
        residual_2d : np.ndarray
            Two-dimensional array (n_t1, n_t2) of residual values.
        t1, t2 : np.ndarray or None
            Time axes (seconds). When both are usable the image is placed on a
            physical t₁/t₂ grid; otherwise the axes fall back to frame indices
            (still labelled t₁/t₂).
        """
        # Levels from the FULL-resolution residuals (block-mean decimation cancels
        # opposite-signed neighbours and would shrink the window vs the histogram/
        # diagonal/scatter diagnostics fed the full surface); rasterize for display.
        full = np.asarray(residual_2d, dtype=float)
        arr = rasterize(full)
        levels = _residual_levels(full)
        self._image_item.setImage(arr, autoLevels=False)
        self._image_item.setLevels(levels)
        self._colorbar.setLevels(low=levels[0], high=levels[1])
        rect = _time_rect(t1, t2)
        if rect is not None:
            self._image_item.setRect(rect)
        # Fit the view to the image extent exactly (padding=0): t₁/t₂ ranges follow
        # the data, so a square residual map gets identical axis ranges and a full fill.
        self._plot.getViewBox().autoRange(padding=0.0)
        self._has_image = True

    def clear_map(self) -> None:
        """Remove any displayed image (so a stale residual is never shown for new input)."""
        self._image_item.clear()
        self._has_image = False

    def has_image(self) -> bool:
        """Return ``True`` after at least one successful ``show_map`` call."""
        return self._has_image
