"""Optional Datashader-backed fast preview.

ImportError-gated: raises on import if datashader/xarray/colorcet aren't
installed. The orchestrator catches this and falls back to matplotlib.

Install via: ``pip install 'xpcsjax[viz-fast]'``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import colorcet  # noqa: F401
    import datashader as ds  # noqa: F401
    import datashader.transfer_functions as tf  # noqa: F401
    import xarray as xr  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Datashader backend requires datashader, xarray, and colorcet. "
        "Install with: pip install 'xpcsjax[viz-fast]'"
    ) from e


def plot_c2_comparison_fast(
    c2_exp: np.ndarray,
    c2_fit: np.ndarray,
    residuals: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    output_file: Path,
    *,
    phi_angle: float | None = None,
    width: int = 1200,
    height: int = 1200,
    percentile_min: float = 1.0,
    percentile_max: float = 99.0,
) -> None:
    """Fast 3-panel render via Datashader, stitched side-by-side via PIL."""
    canvas = ds.Canvas(plot_width=width, plot_height=height)

    def _render(arr, cmap_name, lo, hi):
        da = xr.DataArray(arr, coords={"y": t1, "x": t2}, dims=("y", "x"))
        agg = canvas.raster(da, interpolate="linear")
        return tf.shade(agg, cmap=cmap_name, span=(lo, hi))

    combined = np.concatenate(
        [c2_exp[np.isfinite(c2_exp)].ravel(), c2_fit[np.isfinite(c2_fit)].ravel()]
    )
    if combined.size > 0:
        data_min = float(np.nanpercentile(combined, percentile_min))
        data_max = float(np.nanpercentile(combined, percentile_max))
    else:
        data_min, data_max = 1.0, 1.5
    vmin_shared = max(1.0, data_min)
    vmax_shared = min(1.5, data_max)
    if vmin_shared >= vmax_shared:
        vmax_shared = vmin_shared + 0.5

    finite_r = residuals[np.isfinite(residuals)]
    rmax = float(np.nanpercentile(np.abs(finite_r), 99)) if finite_r.size else 1.0
    if not np.isfinite(rmax) or rmax == 0.0:
        rmax = 1.0

    img_exp = _render(c2_exp, "fire", vmin_shared, vmax_shared)
    img_fit = _render(c2_fit, "fire", vmin_shared, vmax_shared)
    img_res = _render(residuals, "bwr", -rmax, rmax)

    from PIL import Image

    pil_panels = [im.to_pil() for im in (img_exp, img_fit, img_res)]
    total_w = sum(p.width for p in pil_panels)
    total_h = max(p.height for p in pil_panels)
    canvas_img = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 255))
    x_off = 0
    for p in pil_panels:
        canvas_img.paste(p, (x_off, 0))
        x_off += p.width

    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas_img.save(out, "PNG")
