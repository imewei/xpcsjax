"""Interactive PyQtGraph plot widgets for XPCS two-time / residual / overlay views.

All widgets are GUI-process-side only: numpy + Qt + pyqtgraph; no JAX.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
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


def _leading_dim_matches(arr: np.ndarray | None, n: int) -> bool:
    """Return ``True`` iff *arr* is non-None and its leading dim equals *n*.

    Used to reconcile a bundle's optional ``model_c2`` / ``residuals`` against
    ``exp_c2``'s phi count so a mismatched (corrupt/partial) artifact degrades to
    placeholders instead of raising mid-loop.
    """
    return arr is not None and np.asarray(arr).shape[0] == n


# Colormaps mirror the publication NLSQ figures (nlsq_plots.py): a sequential
# "jet" for the C₂ surfaces and a diverging "RdBu_r" centred on zero for the
# residual, so the interactive views match the exported PNGs.
_C2_COLORMAP = "jet"
_RESIDUAL_COLORMAP = "RdBu_r"


def _apply_colormap(image_item: pg.ImageItem, name: str) -> None:
    """Apply a named matplotlib colormap to *image_item* (best-effort).

    Falls back to PyQtGraph's default grayscale if the colormap can't be
    resolved (keeps the GUI usable on an unexpected pyqtgraph/matplotlib build).
    """
    try:
        image_item.setColorMap(pg.colormap.get(name, source="matplotlib"))
    except Exception:  # noqa: BLE001 - colormap availability is environment-dependent
        pass


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


def _time_rect(t1: np.ndarray | None, t2: np.ndarray | None) -> QRectF | None:
    """Map the (t1, t2) time axes to an image rect (x = t₁, y = t₂).

    t₁ is horizontal, t₂ is vertical. Returns ``None`` — so the view falls back
    to pixel-index (frame) coordinates while still being labelled t₁/t₂ — when
    either axis is missing, too short, non-finite, or degenerate.
    """
    if t1 is None or t2 is None:
        return None
    a1 = np.asarray(t1, dtype=float)
    a2 = np.asarray(t2, dtype=float)
    if a1.size < 2 or a2.size < 2:
        return None
    if not (np.isfinite(a1).all() and np.isfinite(a2).all()):
        return None
    x0, x1 = float(np.min(a1)), float(np.max(a1))
    y0, y1 = float(np.min(a2)), float(np.max(a2))
    if x1 <= x0 or y1 <= y0:
        return None
    return QRectF(x0, y0, x1 - x0, y1 - y0)


def _c2_levels(arr: np.ndarray) -> tuple[float, float]:
    """Display window for a C₂ surface: the [1.0, 1.5] band clamped to data.

    Mirrors ``nlsq_plots.py``'s shared vmin/vmax so the bright τ=0 diagonal does
    not saturate the colormap and hide the off-diagonal structure.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (1.0, 1.5)
    vmin = max(1.0, float(np.min(finite)))
    vmax = min(1.5, float(np.max(finite)))
    if vmin >= vmax:
        vmax = vmin + 0.5
    return (vmin, vmax)


def _residual_levels(arr: np.ndarray) -> tuple[float, float]:
    """Symmetric ``[-v, v]`` window (99th pct of |residual|) so RdBu_r centres on 0."""
    finite = np.abs(arr[np.isfinite(arr)])
    vmax = float(np.percentile(finite, 99)) if finite.size > 0 else 1.0
    if vmax == 0.0 or not np.isfinite(vmax):
        vmax = 1.0
    return (-vmax, vmax)


# A plain ``object`` mixin at runtime: subclassing ``QWidget`` here would make the
# concrete views inherit the C++ ``QWidget`` twice and PySide6 segfaults on the
# diamond. For the type-checker we pretend the base is ``QWidget`` so the
# cooperative ``super().resizeEvent`` and the geometry calls (width/height/
# setFixedHeight) resolve — they exist on the real pyqtgraph base at runtime.
if TYPE_CHECKING:
    _SquareBase = QWidget
else:
    _SquareBase = object


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
        arr = rasterize(np.asarray(c2_2d, dtype=float))
        self._image_item.setImage(arr, autoLevels=False)
        self._image_item.setLevels(_c2_levels(arr))
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
        arr = rasterize(np.asarray(residual_2d, dtype=float))
        self._image_item.setImage(arr, autoLevels=False)
        self._image_item.setLevels(_residual_levels(arr))
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


# Cap on points drawn in the Residuals-vs-Fitted scatter. Display-only
# decimation (a full-resolution surface is millions of points and would stall
# the GUI); the histogram and diagonal traces use every finite value, so no
# statistic is ever decimated. A fixed seed keeps the sampled cloud stable.
_SCATTER_MAX_POINTS = 20000


class ResidualHistogramView(_SquareAspectMixin, pg.PlotWidget):
    """Density histogram of residual values with a Normal(μ, σ) overlay.

    Interactive twin of the ``[0, 1]`` panel of the publication
    ``plot_residual_map`` figure (nlsq_plots.py).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.setLabel("bottom", "Residual Value")
        self.setLabel("left", "Density")

    def show_distribution(self, residuals: np.ndarray) -> None:
        """Plot a density histogram of the finite residuals plus a Normal overlay."""
        self.clear()
        finite = np.asarray(residuals, dtype=float).ravel()
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return
        counts, edges = np.histogram(finite, bins=50, density=True)
        # stepMode="center": x (bin edges) is one longer than y (counts).
        self.plot(edges, counts, stepMode="center", fillLevel=0.0, brush=(100, 100, 255, 150))
        mu = float(np.mean(finite))
        sigma = float(np.std(finite))
        if np.isfinite(sigma) and sigma > 0.0:
            x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
            pdf = np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
            self.plot(x, pdf, pen=pg.mkPen("r", width=2))


class DiagonalResidualView(_SquareAspectMixin, pg.PlotWidget):
    """Residual along the t₁ = t₂ diagonal vs time, with a zero reference line.

    Interactive twin of the ``[1, 0]`` panel of ``plot_residual_map``.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.setLabel("bottom", "t₁")
        self.setLabel("left", "Residual")

    def show_diagonal(self, residuals: np.ndarray, t1: np.ndarray | None = None) -> None:
        """Plot ``diag(residuals)`` against the physical (or index) time axis."""
        self.clear()
        diag = np.diag(np.asarray(residuals, dtype=float))
        if diag.size == 0:
            return
        axis = np.asarray(t1, dtype=float) if t1 is not None else np.arange(diag.size, dtype=float)
        if axis.size < diag.size:  # missing/short axis -> fall back to indices
            axis = np.arange(diag.size, dtype=float)
        self.addLine(y=0.0, pen=pg.mkPen("k", style=Qt.PenStyle.DashLine))
        # connect="finite" leaves gaps at masked (NaN) lags rather than bridging them.
        self.plot(axis[: diag.size], diag, pen=pg.mkPen("b", width=1), connect="finite")


class ResidualsVsFittedView(_SquareAspectMixin, pg.PlotWidget):
    """Scatter of residual vs fitted value (heteroscedasticity check).

    Interactive twin of the ``[1, 1]`` panel of ``plot_residual_map``. The point
    cloud is decimated for display only (see ``_SCATTER_MAX_POINTS``).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.setLabel("bottom", "Fitted Value")
        self.setLabel("left", "Residual")

    def show_scatter(self, fitted: np.ndarray, residuals: np.ndarray) -> None:
        """Scatter finite (fitted, residual) pairs, decimated to a display cap."""
        self.clear()
        xf = np.asarray(fitted, dtype=float).ravel()
        yr = np.asarray(residuals, dtype=float).ravel()
        n = min(xf.size, yr.size)
        xf, yr = xf[:n], yr[:n]
        mask = np.isfinite(xf) & np.isfinite(yr)
        xf, yr = xf[mask], yr[mask]
        if xf.size == 0:
            return
        if xf.size > _SCATTER_MAX_POINTS:
            idx = np.random.default_rng(0).choice(xf.size, _SCATTER_MAX_POINTS, replace=False)
            xf, yr = xf[idx], yr[idx]
        self.addLine(y=0.0, pen=pg.mkPen("r", style=Qt.PenStyle.DashLine))
        self.addItem(pg.ScatterPlotItem(x=xf, y=yr, size=2, pen=None, brush=(100, 100, 255, 40)))


class _PhiSection(QWidget):
    """One phi-angle row: Exp | Fitted | Residual live maps + interactive diagnostics.

    A self-contained section so :class:`PhiResultsGrid` can build/replace them
    one per angle. Below the maps sits a row of three interactive residual
    diagnostics — Residual Distribution | Diagonal Residuals | Residuals vs
    Fitted — mirroring the panels of the publication ``plot_residual_map``
    figure. Missing fitted/residual surfaces degrade to labelled placeholders;
    the section never raises.
    """

    _MAP_MIN_HEIGHT = 240
    _DIAG_MIN_HEIGHT = 240

    def __init__(
        self,
        exp_2d: np.ndarray,
        fitted_2d: np.ndarray | None,
        residual_2d: np.ndarray | None,
        phi_deg: float,
        t1: np.ndarray | None = None,
        t2: np.ndarray | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._has_fitted = fitted_2d is not None
        self._has_residual = residual_2d is not None
        # The diagnostics row needs both the residual surface and the fitted
        # surface (Residuals-vs-Fitted); exp-only views show a placeholder.
        self._has_diagnostics = residual_2d is not None and fitted_2d is not None
        self._t1 = t1
        self._t2 = t2

        layout = QVBoxLayout(self)
        header = QLabel(f"φ = {phi_deg:.3f}°")
        header.setObjectName("phi_section_header")
        layout.addWidget(header)

        maps_row = QHBoxLayout()
        maps_row.addWidget(self._titled_map("Experimental C₂", exp_2d, residual=False))
        maps_row.addWidget(self._titled_map("Fitted C₂", fitted_2d, residual=False))
        maps_row.addWidget(self._titled_map("Residuals", residual_2d, residual=True))
        layout.addLayout(maps_row)

        layout.addWidget(self._diagnostics_row(fitted_2d, residual_2d))

    @staticmethod
    def _titled(title: str, widget: QWidget) -> QWidget:
        """Wrap *widget* in a column with a header label (the maps/diagnostics layout)."""
        col = QWidget()
        col_layout = QVBoxLayout(col)
        col_layout.setContentsMargins(0, 0, 0, 0)
        col_layout.addWidget(QLabel(title))
        col_layout.addWidget(widget)
        return col

    def _titled_map(self, title: str, data: np.ndarray | None, *, residual: bool) -> QWidget:
        """Build a titled column holding a map view (or a placeholder when *data* is None)."""
        if data is None:
            placeholder = QLabel("(not available)")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setMinimumHeight(self._MAP_MIN_HEIGHT)
            return self._titled(title, placeholder)
        view: ResidualMapView | TwoTimeMapView = ResidualMapView() if residual else TwoTimeMapView()
        view.setMinimumHeight(self._MAP_MIN_HEIGHT)
        view.show_map(np.asarray(data), self._t1, self._t2)
        return self._titled(title, view)

    def _diagnostics_row(
        self, fitted_2d: np.ndarray | None, residual_2d: np.ndarray | None
    ) -> QWidget:
        """Build the interactive residual-diagnostics row (3 plots), or a placeholder.

        Replaces the per-angle static residuals PNG with live Residual
        Distribution | Diagonal Residuals | Residuals vs Fitted plots. Requires
        both the residual and fitted surfaces; an exp-only section shows a
        single placeholder instead.
        """
        container = QWidget()
        container.setObjectName("phi_section_diagnostics")
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        if residual_2d is None or fitted_2d is None:
            placeholder = QLabel("(residual diagnostics unavailable — no fitted surface)")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setMinimumHeight(self._DIAG_MIN_HEIGHT)
            row.addWidget(placeholder)
            return container

        residual = np.asarray(residual_2d)
        fitted = np.asarray(fitted_2d)

        hist = ResidualHistogramView()
        hist.show_distribution(residual)
        diagonal = DiagonalResidualView()
        diagonal.show_diagonal(residual, self._t1)
        scatter = ResidualsVsFittedView()
        scatter.show_scatter(fitted, residual)

        for title, view in (
            ("Residual Distribution", hist),
            ("Diagonal Residuals", diagonal),
            ("Residuals vs Fitted", scatter),
        ):
            view.setMinimumHeight(self._DIAG_MIN_HEIGHT)
            row.addWidget(self._titled(title, view))
        return container


class PhiResultsGrid(QWidget):
    """Scrollable per-phi results grid sized to ``n_phi`` at load time.

    One :class:`_PhiSection` per phi angle — each showing the experimental /
    fitted / residual two-time maps (live, interactive) plus an interactive
    residual-diagnostics row. No spinbox: every angle is laid out at once and
    the whole grid scrolls.

    Parameters
    ----------
    parent : QWidget or None
        Optional parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        # Keep the viewport width constant. Every tile is squared by
        # _SquareAspectMixin (height := width on resize); if the vertical
        # scrollbar could toggle, each toggle would change the width, re-pin all
        # tile heights, change the content height, and re-toggle the bar — an
        # oscillation that left tiles rectangular mid-flight (stretched image,
        # wide letterbox margins). Pinning the bar on (and the horizontal one
        # off) fixes the width, so the square-tile loop converges immediately.
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._container = QWidget()
        self._vbox = QVBoxLayout(self._container)
        self._scroll.setWidget(self._container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._scroll)

        self._bundle: VizBundle | None = None
        self._sections: list[_PhiSection] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def set_bundle(self, bundle: VizBundle | None) -> None:
        """Rebuild the grid from *bundle* (one interactive section per phi angle).

        Parameters
        ----------
        bundle : VizBundle or None
            Fit data bundle. ``None`` clears the grid.
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

        # One section per phi angle, sized to n_phi.
        for i in range(n_phi):
            fitted = model_c2[i] if model_c2 is not None else None
            residual = residuals[i] if residuals is not None else None
            section = _PhiSection(
                exp_2d=bundle.exp_c2[i],
                fitted_2d=fitted,
                residual_2d=residual,
                phi_deg=float(phi_angles[i]),
                t1=bundle.t1,
                t2=bundle.t2,
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
        """Remove all section widgets so a prior run's grid never lingers."""
        while self._vbox.count():
            item = self._vbox.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._sections = []
