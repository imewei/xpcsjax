"""Low-level helpers and constants shared across the plots subpackage.

Colormaps, level functions, and utility functions — no Qt widget classes.
All GUI-process-side only: numpy + pyqtgraph; no JAX.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF

# Colormaps mirror the publication NLSQ figures (nlsq_plots.py): a sequential
# "jet" for the C₂ surfaces and a diverging "RdBu_r" centred on zero for the
# residual, so the interactive views match the exported PNGs.
_C2_COLORMAP = "jet"
_RESIDUAL_COLORMAP = "RdBu_r"

# Cap on points drawn in the Residuals-vs-Fitted scatter. Display-only
# decimation (a full-resolution surface is millions of points and would stall
# the GUI); the histogram and diagonal traces use every finite value, so no
# statistic is ever decimated. A fixed seed keeps the sampled cloud stable.
_SCATTER_MAX_POINTS = 20000


def _leading_dim_matches(arr: np.ndarray | None, n: int) -> bool:
    """Return ``True`` iff *arr* is non-None and its leading dim equals *n*.

    Used to reconcile a bundle's optional ``model_c2`` / ``residuals`` against
    ``exp_c2``'s phi count so a mismatched (corrupt/partial) artifact degrades to
    placeholders instead of raising mid-loop.
    """
    if arr is None:
        return False
    a = np.asarray(arr)
    return a.ndim > 0 and a.shape[0] == n


def _resolve_colormap(name: str) -> pg.ColorMap | None:
    """Resolve a named matplotlib colormap (best-effort); ``None`` if unavailable.

    Shared by every colormap consumer (the image item AND its colorbar) so
    "colormap availability is environment-dependent" is guarded exactly once —
    an unguarded second call site would crash widget construction on the same
    environments this degrades gracefully on.
    """
    try:
        return pg.colormap.get(name, source="matplotlib")
    except Exception:  # noqa: BLE001 - colormap availability is environment-dependent
        return None


def _apply_colormap(image_item: pg.ImageItem, name: str) -> None:
    """Apply a named matplotlib colormap to *image_item* (best-effort).

    Falls back to PyQtGraph's default grayscale if the colormap can't be
    resolved (keeps the GUI usable on an unexpected pyqtgraph/matplotlib build).
    """
    cmap = _resolve_colormap(name)
    if cmap is not None:
        image_item.setColorMap(cmap)


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
