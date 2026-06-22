"""Scrollable per-phi results grid.

``_PhiSection`` and ``PhiResultsGrid`` — one interactive section per phi angle
showing experimental / fitted / residual two-time maps plus residual diagnostics.
All GUI-process-side only: numpy + Qt + pyqtgraph; no JAX.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .helpers import _leading_dim_matches
from .maps import ResidualMapView, TwoTimeMapView
from .residuals import DiagonalResidualView, ResidualHistogramView, ResidualsVsFittedView

if TYPE_CHECKING:
    from xpcsjax.gui.viz_bundle import VizBundle


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
