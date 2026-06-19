"""Interactive PyQtGraph plot widgets for XPCS two-time / residual / overlay views.

All widgets are GUI-process-side only: numpy + Qt + pyqtgraph; no JAX.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QSpinBox, QVBoxLayout, QWidget

from xpcsjax.gui.views.raster import rasterize

if TYPE_CHECKING:
    from xpcsjax.gui.viz_bundle import VizBundle


class TwoTimeMapView(pg.GraphicsLayoutWidget):
    """Display a two-time correlation matrix as a pan/zoom-able image.

    Parameters
    ----------
    parent : QWidget or None
        Optional parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._plot = self.addPlot()
        self._image_item = pg.ImageItem()
        self._plot.addItem(self._image_item)
        self._has_image = False

    def show_map(self, c2_2d: np.ndarray) -> None:
        """Rasterize and display a 2-D correlation matrix.

        Parameters
        ----------
        c2_2d : np.ndarray
            Two-dimensional array (n_t, n_t) of correlation values.
        """
        arr = rasterize(np.asarray(c2_2d, dtype=float))
        self._image_item.setImage(arr)
        self._has_image = True

    def clear_map(self) -> None:
        """Remove any displayed image (so a stale map is never shown for new input)."""
        self._image_item.clear()
        self._has_image = False

    def has_image(self) -> bool:
        """Return ``True`` after at least one successful ``show_map`` call."""
        return self._has_image


class ResidualMapView(pg.GraphicsLayoutWidget):
    """Display a residual map as a pan/zoom-able image.

    Parameters
    ----------
    parent : QWidget or None
        Optional parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._plot = self.addPlot()
        self._image_item = pg.ImageItem()
        self._plot.addItem(self._image_item)
        self._has_image = False

    def show_map(self, residual_2d: np.ndarray) -> None:
        """Rasterize and display a 2-D residual matrix.

        Parameters
        ----------
        residual_2d : np.ndarray
            Two-dimensional array (n_t, n_t) of residual values.
        """
        arr = rasterize(np.asarray(residual_2d, dtype=float))
        self._image_item.setImage(arr)
        self._has_image = True

    def clear_map(self) -> None:
        """Remove any displayed image (so a stale residual is never shown for new input)."""
        self._image_item.clear()
        self._has_image = False

    def has_image(self) -> bool:
        """Return ``True`` after at least one successful ``show_map`` call."""
        return self._has_image


class PerAngleOverlayView(pg.PlotWidget):
    """Line plots of per-angle g2 values (experimental and optional model).

    Parameters
    ----------
    parent : QWidget or None
        Optional parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._curves: list[pg.PlotDataItem] = []

    def show_overlay(
        self,
        phi_angles: np.ndarray,
        exp_g2: np.ndarray,
        model_g2: np.ndarray | None = None,
    ) -> None:
        """Plot experimental (and optional model) g2 vs phi angles.

        Parameters
        ----------
        phi_angles : np.ndarray
            Phi angle values (degrees), shape (n_phi,).
        exp_g2 : np.ndarray
            Experimental per-angle g2 scalars, shape (n_phi,).
        model_g2 : np.ndarray or None
            Model per-angle g2 scalars, shape (n_phi,), or ``None``.
        """
        self.clear()
        self._curves = []

        phi = np.asarray(phi_angles, dtype=float)
        exp = np.asarray(exp_g2, dtype=float)
        curve_exp = self.plot(phi, exp, pen=pg.mkPen("b", width=1.5), name="exp")
        self._curves.append(curve_exp)

        if model_g2 is not None:
            mdl = np.asarray(model_g2, dtype=float)
            curve_mdl = self.plot(phi, mdl, pen=pg.mkPen("r", width=1.5), name="model")
            self._curves.append(curve_mdl)

    def curve_count(self) -> int:
        """Return the number of curves currently plotted (0, 1, or 2)."""
        return len(self._curves)


class ResultPlots(QWidget):
    """Composite widget: two-time map + residual map + per-angle overlay + phi spinbox.

    ``set_bundle`` drives all three sub-views from a single :class:`VizBundle`.
    The phi spinbox selects which phi slice is shown in the map views.

    Parameters
    ----------
    parent : QWidget or None
        Optional parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        self._two_time = TwoTimeMapView()
        self._residual = ResidualMapView()
        self._overlay = PerAngleOverlayView()

        self._spinbox = QSpinBox()
        self._spinbox.setMinimum(0)
        self._spinbox.setMaximum(0)
        self._spinbox.valueChanged.connect(self._on_phi_changed)

        self._bundle: VizBundle | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(self._spinbox)
        layout.addWidget(self._two_time)
        layout.addWidget(self._residual)
        layout.addWidget(self._overlay)
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_bundle(self, bundle: VizBundle | None) -> None:
        """Load a :class:`VizBundle` into all plot views.

        Parameters
        ----------
        bundle : VizBundle or None
            Data bundle from the fit.  ``None`` clears all views.
        """
        self._bundle = bundle

        if bundle is None:
            self._spinbox.setMaximum(0)
            self._spinbox.setValue(0)
            # Clear every sub-view so a prior run's plots don't linger.
            self._two_time.clear_map()
            self._residual.clear_map()
            self._overlay.clear()
            return

        n_phi = bundle.exp_c2.shape[0]
        # Block signals while resetting range so _on_phi_changed doesn't
        # fire until we call it explicitly with the initial slice.
        self._spinbox.blockSignals(True)
        self._spinbox.setMinimum(0)
        self._spinbox.setMaximum(max(0, n_phi - 1))
        self._spinbox.setValue(0)
        self._spinbox.blockSignals(False)

        self._render_phi(0)

    def phi_count(self) -> int:
        """Return number of phi slices in the current bundle (0 when no bundle)."""
        if self._bundle is None:
            return 0
        return int(self._bundle.exp_c2.shape[0])

    def two_time(self) -> TwoTimeMapView:
        """Return the :class:`TwoTimeMapView` sub-widget."""
        return self._two_time

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _on_phi_changed(self, idx: int) -> None:
        """Slot: re-render map views when the spinbox value changes."""
        if self._bundle is None:
            return
        self._render_phi(idx)

    def _render_phi(self, idx: int) -> None:
        """Render all views for phi slice ``idx``.

        Parameters
        ----------
        idx : int
            Phi index into the bundle's leading dimension.
        """
        bundle = self._bundle
        if bundle is None:
            return

        n_phi = bundle.exp_c2.shape[0]
        idx = max(0, min(idx, n_phi - 1))

        # --- Two-time map ---
        self._two_time.show_map(bundle.exp_c2[idx])

        # --- Residual map (clear if absent, so a prior run's residual never lingers) ---
        if bundle.residuals is not None:
            self._residual.show_map(bundle.residuals[idx])
        else:
            self._residual.clear_map()

        # --- Per-angle overlay ---
        # Scalar per φ = mean of first superdiagonal (τ=dt).
        phi_angles = bundle.phi_angles
        if phi_angles is None:
            phi_angles = np.arange(n_phi, dtype=float)

        exp_g2 = np.array(
            [float(np.mean(np.diagonal(bundle.exp_c2[i], offset=1))) for i in range(n_phi)]
        )
        model_g2: np.ndarray | None = None
        if bundle.model_c2 is not None:
            model_g2 = np.array(
                [float(np.mean(np.diagonal(bundle.model_c2[i], offset=1))) for i in range(n_phi)]
            )

        self._overlay.show_overlay(phi_angles, exp_g2, model_g2)
