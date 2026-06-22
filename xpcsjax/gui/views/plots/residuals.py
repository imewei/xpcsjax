"""Residual diagnostic plot views.

``ResidualHistogramView``, ``DiagonalResidualView``, and ``ResidualsVsFittedView``
— interactive residual diagnostics mirroring the panels of the publication
``plot_residual_map`` figure. All GUI-process-side only: numpy + Qt + pyqtgraph;
no JAX.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from .helpers import _SCATTER_MAX_POINTS
from .squares import _SquareAspectMixin


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
