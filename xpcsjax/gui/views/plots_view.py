"""Facade re-exporting the GUI plot widgets (now in the plots/ subpackage)."""

from xpcsjax.gui.views.plots.grid import PhiResultsGrid, _PhiSection
from xpcsjax.gui.views.plots.helpers import (
    _SCATTER_MAX_POINTS,
    _c2_levels,
    _residual_levels,
    _time_rect,
)
from xpcsjax.gui.views.plots.maps import ResidualMapView, TwoTimeMapView
from xpcsjax.gui.views.plots.residuals import (
    DiagonalResidualView,
    ResidualHistogramView,
    ResidualsVsFittedView,
)
from xpcsjax.gui.views.plots.squares import _SquareAspectMixin

__all__ = [
    "PhiResultsGrid",
    "TwoTimeMapView",
    "ResidualMapView",
    "ResidualHistogramView",
    "DiagonalResidualView",
    "ResidualsVsFittedView",
    "_SCATTER_MAX_POINTS",
    "_c2_levels",
    "_residual_levels",
    "_time_rect",
    "_SquareAspectMixin",
    "_PhiSection",
]
