"""Interactive PyQtGraph plot widgets for XPCS two-time / residual / overlay views.

All widgets are GUI-process-side only: numpy + Qt + pyqtgraph; no JAX.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from xpcsjax.gui.views.raster import rasterize

if TYPE_CHECKING:
    from xpcsjax.gui.viz_bundle import VizBundle


def _superdiag_mean(matrix: np.ndarray) -> float:
    """Mean of the first superdiagonal (the τ=dt lag) of a two-time matrix.

    Returns ``nan`` for a matrix too small to have a first superdiagonal
    (``shape[1] < 2``) — e.g. a degenerate ``(1, 1)`` slice. Guarding here avoids
    ``np.mean`` of an empty slice, which would emit a ``RuntimeWarning`` and yield
    ``nan`` implicitly; ``nan`` is the correct "no τ=dt data" sentinel for the
    overlay (pyqtgraph renders it as a gap).
    """
    diag = np.diagonal(np.asarray(matrix), offset=1)
    if diag.size == 0:
        return float("nan")
    return float(np.mean(diag))


def _leading_dim_matches(arr: np.ndarray | None, n: int) -> bool:
    """Return ``True`` iff *arr* is non-None and its leading dim equals *n*.

    Used to reconcile a bundle's optional ``model_c2`` / ``residuals`` against
    ``exp_c2``'s phi count so a mismatched (corrupt/partial) artifact degrades to
    placeholders instead of raising mid-loop.
    """
    return arr is not None and np.asarray(arr).shape[0] == n


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


def find_diagnostics_png(result_dir: str | Path | None, phi_idx: int) -> Path | None:
    """Locate the per-angle residual-diagnostics PNG for slice ``phi_idx``.

    The fit writes ``residuals_phi_{idx:03d}_{deg:.3f}deg.png`` into the run's
    plot directory (``nlsq_plots.py``). The exact ``deg`` suffix is not known
    here, so we glob by the zero-padded index and return the first match. The
    search is non-recursive and bounded to the two known locations — the
    ``<result_dir>/plots`` directory and the run dir itself — so it never walks
    a large tree on the GUI thread.

    Parameters
    ----------
    result_dir : str or Path or None
        The run's result directory. ``None`` returns ``None``.
    phi_idx : int
        Phi slice index (matched against the ``phi_{idx:03d}`` filename token).

    Returns
    -------
    pathlib.Path or None
        Path to the diagnostics PNG, or ``None`` when absent/unreadable.
    """
    if result_dir is None:
        return None
    root = Path(result_dir)
    pattern = f"residuals_phi_{phi_idx:03d}_*.png"
    # The fit writes PNGs under <result_dir>/plots/; also accept them directly in
    # the run dir. Non-recursive glob on each — no rglob tree-walk on the GUI thread.
    for search_dir in (root / "plots", root):
        try:
            if not search_dir.is_dir():
                continue
            matches = sorted(search_dir.glob(pattern))
        except OSError:
            continue
        if matches:
            return matches[0]
    return None


class _PhiSection(QWidget):
    """One phi-angle row: Exp | Fitted | Residual live maps + diagnostics PNG.

    A self-contained section so :class:`PhiResultsGrid` can build/replace them
    one per angle. Missing fitted/residual surfaces or a missing diagnostics PNG
    each degrade to a labelled placeholder — the section never raises.
    """

    _MAP_MIN_HEIGHT = 240
    _PNG_MAX_WIDTH = 920

    def __init__(
        self,
        exp_2d: np.ndarray,
        fitted_2d: np.ndarray | None,
        residual_2d: np.ndarray | None,
        phi_deg: float,
        diagnostics_png: Path | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._has_fitted = fitted_2d is not None
        self._has_residual = residual_2d is not None
        self._has_png = diagnostics_png is not None

        layout = QVBoxLayout(self)
        header = QLabel(f"φ = {phi_deg:.3f}°")
        header.setObjectName("phi_section_header")
        layout.addWidget(header)

        maps_row = QHBoxLayout()
        maps_row.addWidget(self._titled_map("Experimental C₂", exp_2d, residual=False))
        maps_row.addWidget(self._titled_map("Fitted C₂", fitted_2d, residual=False))
        maps_row.addWidget(self._titled_map("Residuals", residual_2d, residual=True))
        layout.addLayout(maps_row)

        layout.addWidget(self._diagnostics_widget(diagnostics_png))

    def _titled_map(self, title: str, data: np.ndarray | None, *, residual: bool) -> QWidget:
        """Build a titled column holding a map view (or a placeholder when *data* is None)."""
        col = QWidget()
        col_layout = QVBoxLayout(col)
        col_layout.setContentsMargins(0, 0, 0, 0)
        col_layout.addWidget(QLabel(title))
        if data is None:
            placeholder = QLabel("(not available)")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setMinimumHeight(self._MAP_MIN_HEIGHT)
            col_layout.addWidget(placeholder)
            return col
        view: ResidualMapView | TwoTimeMapView = ResidualMapView() if residual else TwoTimeMapView()
        view.setMinimumHeight(self._MAP_MIN_HEIGHT)
        view.show_map(np.asarray(data))
        col_layout.addWidget(view)
        return col

    def _diagnostics_widget(self, png: Path | None) -> QWidget:
        """Return a QLabel showing the diagnostics PNG, or a placeholder if absent."""
        label = QLabel()
        label.setObjectName("phi_section_diagnostics")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap(str(png)) if png is not None else QPixmap()
        if png is None or pixmap.isNull():
            label.setText("(residual diagnostics image not found)")
            self._has_png = False
            return label
        if pixmap.width() > self._PNG_MAX_WIDTH:
            pixmap = pixmap.scaledToWidth(
                self._PNG_MAX_WIDTH, Qt.TransformationMode.SmoothTransformation
            )
        label.setPixmap(pixmap)
        return label


class PhiResultsGrid(QWidget):
    """Scrollable per-phi results grid sized to ``n_phi`` at load time.

    A summary g₂-vs-φ overlay sits above one :class:`_PhiSection` per phi angle —
    each showing the experimental / fitted / residual two-time maps (live,
    interactive) and the embedded residual-diagnostics PNG the fit produced. No
    spinbox: every angle is laid out at once and the whole grid scrolls.

    Parameters
    ----------
    parent : QWidget or None
        Optional parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._container = QWidget()
        self._vbox = QVBoxLayout(self._container)
        self._scroll.setWidget(self._container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._scroll)

        self._bundle: VizBundle | None = None
        self._sections: list[_PhiSection] = []
        self._overlay: PerAngleOverlayView | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def set_bundle(self, bundle: VizBundle | None, result_dir: str | Path | None = None) -> None:
        """Rebuild the grid from *bundle* (and locate diagnostics PNGs under *result_dir*).

        Parameters
        ----------
        bundle : VizBundle or None
            Fit data bundle. ``None`` clears the grid.
        result_dir : str or Path or None
            Run directory used to locate per-angle ``residuals_phi_*.png`` images.
        """
        self._clear()
        self._bundle = bundle
        if bundle is None:
            return

        n_phi = int(bundle.exp_c2.shape[0])
        # Defensive length reconciliation: a malformed/partial artifact whose
        # optional arrays disagree with exp_c2's leading dim must degrade to
        # placeholders, never raise an IndexError mid-loop. (The in-package
        # pipeline always agrees; this guards hand-edited / external bundles.)
        model_c2 = bundle.model_c2 if _leading_dim_matches(bundle.model_c2, n_phi) else None
        residuals = bundle.residuals if _leading_dim_matches(bundle.residuals, n_phi) else None
        phi_angles = bundle.phi_angles
        if phi_angles is None or np.asarray(phi_angles).shape[0] != n_phi:
            phi_angles = np.arange(n_phi, dtype=float)
        phi_angles = np.asarray(phi_angles, dtype=float)

        # Summary overlay (g2 vs all phi) pinned at the top.
        exp_g2 = np.array([_superdiag_mean(bundle.exp_c2[i]) for i in range(n_phi)])
        model_g2: np.ndarray | None = None
        if model_c2 is not None:
            model_g2 = np.array([_superdiag_mean(model_c2[i]) for i in range(n_phi)])
        overlay = PerAngleOverlayView()
        overlay.setMinimumHeight(200)
        overlay.show_overlay(phi_angles, exp_g2, model_g2)
        self._vbox.addWidget(overlay)
        self._overlay = overlay

        # One section per phi angle, sized to n_phi.
        for i in range(n_phi):
            fitted = model_c2[i] if model_c2 is not None else None
            residual = residuals[i] if residuals is not None else None
            section = _PhiSection(
                exp_2d=bundle.exp_c2[i],
                fitted_2d=fitted,
                residual_2d=residual,
                phi_deg=float(phi_angles[i]),
                diagnostics_png=find_diagnostics_png(result_dir, i),
            )
            self._vbox.addWidget(section)
            self._sections.append(section)

        self._vbox.addStretch(1)

    def section_count(self) -> int:
        """Return the number of per-phi sections currently built (0 when no bundle)."""
        return len(self._sections)

    def phi_count(self) -> int:
        """Return number of phi slices in the current bundle (0 when no bundle)."""
        if self._bundle is None:
            return 0
        return int(self._bundle.exp_c2.shape[0])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _clear(self) -> None:
        """Remove all overlay + section widgets so a prior run's grid never lingers."""
        while self._vbox.count():
            item = self._vbox.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._sections = []
        self._overlay = None
