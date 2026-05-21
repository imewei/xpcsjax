"""NLSQ visualization for xpcsjax."""

from xpcsjax.viz.diagnostics import (
    DiagonalOverlayResult,
    compute_diagonal_overlay_stats,
)
from xpcsjax.viz.nlsq_plots import (
    generate_nlsq_plots,
    plot_nlsq_fit,
    plot_residual_map,
    plot_simulated_data,
)

__all__ = [
    "DiagonalOverlayResult",
    "compute_diagonal_overlay_stats",
    "generate_nlsq_plots",
    "plot_nlsq_fit",
    "plot_residual_map",
    "plot_simulated_data",
]
