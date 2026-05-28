"""NLSQ visualization for xpcsjax.

Lazy public surface — symbols are imported on first access via ``__getattr__``.
This mirrors the top-level ``xpcsjax/__init__.py`` pattern and keeps matplotlib
out of the import chain unless a user actually requests a viz function.

Later tasks add entries to ``_LAZY_EXPORTS`` and ``__all__`` as they implement
each public symbol (Task 6 → ``plot_nlsq_fit``, Task 7 → ``plot_residual_map``,
Task 8 → ``plot_simulated_data``, Task 11 → ``generate_nlsq_plots``,
Task 15 → ``DiagonalOverlayResult`` and ``compute_diagonal_overlay_stats``).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# Static visibility for Pyright (reportUnsupportedDunderAll): the public symbols
# are resolved lazily via __getattr__ at runtime, so import them here under
# TYPE_CHECKING — seen by type checkers, not imported at runtime (keeps
# matplotlib out of the import chain). Mirrors the top-level xpcsjax/__init__.py.
if TYPE_CHECKING:
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

# Map public symbol → submodule that defines it. Each later task appends here.
_LAZY_EXPORTS: dict[str, str] = {
    "DiagonalOverlayResult": "xpcsjax.viz.diagnostics",
    "compute_diagonal_overlay_stats": "xpcsjax.viz.diagnostics",
    "generate_nlsq_plots": "xpcsjax.viz.nlsq_plots",
    "plot_nlsq_fit": "xpcsjax.viz.nlsq_plots",
    "plot_residual_map": "xpcsjax.viz.nlsq_plots",
    "plot_simulated_data": "xpcsjax.viz.nlsq_plots",
}

# Literal list — Pyright reportUnsupportedDunderAll requires this stays a literal.
# Later tasks append symbol names here as they're implemented.
__all__: list[str] = [
    "DiagonalOverlayResult",
    "compute_diagonal_overlay_stats",
    "generate_nlsq_plots",
    "plot_nlsq_fit",
    "plot_residual_map",
    "plot_simulated_data",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name])
        attr = getattr(module, name)
        globals()[name] = attr  # cache for next access
        return attr
    raise AttributeError(f"module 'xpcsjax.viz' has no attribute {name!r}")
