"""NLSQ fit visualization and artifact serialization.

Symbols defined here are wired into ``xpcsjax.viz``'s lazy export map by later
tasks (Task 2 onward).
"""

from __future__ import annotations

import numpy as np


def _resolve_color_limits(
    matrix: np.ndarray,
    percentile_min: float = 1.0,
    percentile_max: float = 99.0,
) -> tuple[float, float]:
    """Percentile-based color limits with NaN/empty/flat fallbacks.

    Returns ``(1.0, 1.5)`` when the input is empty or all-NaN. Widens flat data
    to ``(vmin, vmin + 1.0)`` so matplotlib's imshow doesn't render a blank
    image with an invalid colorbar.
    """
    if matrix.size == 0 or not np.any(np.isfinite(matrix)):
        return 1.0, 1.5
    vmin = float(np.nanpercentile(matrix, percentile_min))
    vmax = float(np.nanpercentile(matrix, percentile_max))
    if not np.isfinite(vmin):
        vmin = 1.0
    if not np.isfinite(vmax):
        vmax = 1.5
    if vmin >= vmax:
        vmax = vmin + 1.0
    return vmin, vmax
