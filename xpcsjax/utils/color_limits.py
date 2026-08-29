"""Shared color-limit helper for residual heatmap rendering.

Single source of truth for the symmetric "±99th-percentile-of-|residual|"
window used to center a diverging (``RdBu_r``) colormap on zero. Reused by
the matplotlib backend (:mod:`xpcsjax.viz.nlsq_plots`), the Datashader
backend (:mod:`xpcsjax.viz.datashader_backend`), and the PyQtGraph GUI
backend (:mod:`xpcsjax.gui.views.plots.helpers`) so all three render the
same residual color scale for the same data. Lives in ``xpcsjax.utils``
(not ``xpcsjax.viz``) because it must be importable from the GUI process,
which is JAX-free and cannot import :mod:`xpcsjax.viz` (that package
imports ``jax.numpy`` at module scope).
"""

from __future__ import annotations

import numpy as np


def symmetric_residual_limit(residual: np.ndarray, percentile: float = 99.0) -> tuple[float, float]:
    """Symmetric ``(-v, v)`` color limits for a residual surface.

    ``v`` is the *percentile*-th percentile of ``|residual|`` computed over
    finite values only, so a diverging colormap's midpoint always maps to
    zero. Falls back to ``v = 1.0`` when there are no finite values, or when
    the computed percentile is zero or non-finite (a degenerate all-zero
    residual).
    """
    finite = residual[np.isfinite(residual)]
    vmax = float(np.percentile(np.abs(finite), percentile)) if finite.size > 0 else 1.0
    if vmax == 0.0 or not np.isfinite(vmax):
        vmax = 1.0
    return (-vmax, vmax)
