"""Datashader backend for fast C2 heatmap visualization.

Provides high-performance CPU-optimized heatmap rendering using Datashader,
delivering 5-10x per-call speedup over matplotlib for C2 correlation data.
With multiprocessing fan-out across angles (see :func:`generate_nlsq_plots`)
the cumulative speedup on a many-core box reaches 50-200x.

Why this matters: a single XPCS angle can carry 10^7 - 10^8 c2 samples (e.g.
a 10000 × 10000 t1×t2 surface). Pure matplotlib ``imshow`` on raw data of
that size is dominated by canvas-level resampling and is too slow for batch
sweeps. Datashader rasterizes the raw array to a fixed-resolution image
(e.g. 800x800) on the CPU, then matplotlib only ever sees the pre-rasterized
image — keeping the matplotlib path tiny while preserving annotation
quality (colorbars, axes, titles).

Optional dependency — only import this module if Datashader is installed:

    pip install 'xpcsjax[viz-fast]'

The orchestrator in :mod:`xpcsjax.viz.nlsq_plots` checks
``DATASHADER_AVAILABLE`` and degrades to the matplotlib path on missing
deps; users who never set ``use_datashader=True`` never pay the import cost.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import datashader as ds
    import datashader.transfer_functions as tf
    import xarray as xr
    from PIL import Image
except ImportError as e:
    raise ImportError(
        "Datashader backend requires datashader, xarray, and Pillow. "
        "Install with: pip install 'xpcsjax[viz-fast]'"
    ) from e

from xpcsjax.utils.logging import get_logger
from xpcsjax.utils.path_validation import validate_plot_save_path

logger = get_logger(__name__)


class DatashaderRenderer:
    """Fast heatmap rendering using Datashader.

    Uses Datashader's CPU rasterization pipeline to convert 2D gridded data
    (C2 correlation surfaces) into RGB images. The output is a PIL Image
    that callers can save directly or hand to matplotlib for annotation.

    Examples
    --------
    >>> renderer = DatashaderRenderer(width=800, height=800)
    >>> img = renderer.rasterize_heatmap(c2_data, t2_coords, t1_coords)
    >>> img.save("output.png")  # direct PIL save
    """

    def __init__(self, width: int = 800, height: int = 800) -> None:
        self.width = width
        self.height = height

    def rasterize_heatmap(
        self,
        data: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        cmap: str = "jet",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> Image.Image:
        """Rasterize 2D gridded data to a PIL Image using Datashader.

        Parameters
        ----------
        data
            2D array, shape ``(n_y, n_x)``. For C2 data: pass ``c2.T`` to
            swap axes for correct display.
        x_coords, y_coords
            Coordinate arrays of length ``n_x`` and ``n_y``. For C2 data:
            ``x_coords = t1``, ``y_coords = t2``.
        cmap
            Matplotlib colormap name (converted to a Datashader-friendly
            hex-color list internally).
        vmin, vmax
            Color-scale limits. If ``None`` they default to data min/max.
        """
        if data.shape[0] != len(y_coords):
            raise ValueError(
                f"data y-dim ({data.shape[0]}) doesn't match y_coords length ({len(y_coords)})"
            )
        if data.shape[1] != len(x_coords):
            raise ValueError(
                f"data x-dim ({data.shape[1]}) doesn't match x_coords length ({len(x_coords)})"
            )

        xr_data = xr.DataArray(
            data,
            coords={"y": y_coords, "x": x_coords},
            dims=["y", "x"],
            name="intensity",
        )

        # NaN-tolerant coordinate range computation.
        x_finite = x_coords[np.isfinite(x_coords)]
        y_finite = y_coords[np.isfinite(y_coords)]
        if x_finite.size == 0 or y_finite.size == 0:
            raise ValueError("Cannot rasterize: all coordinate values are NaN")

        canvas = ds.Canvas(
            plot_width=self.width,
            plot_height=self.height,
            x_range=(float(x_finite.min()), float(x_finite.max())),
            y_range=(float(y_finite.min()), float(y_finite.max())),
        )

        # canvas.quadmesh() resamples gridded data to canvas resolution.
        # canvas.raster() was removed in Datashader ≥ 0.15; quadmesh() is the
        # supported API for regularly-spaced xr.DataArray inputs.
        agg = canvas.quadmesh(xr_data, x="x", y="y", agg=ds.mean("intensity"))
        cmap_obj = self._get_colormap(cmap)

        if vmin is None or vmax is None:
            data_finite = data[np.isfinite(data)]
            if data_finite.size == 0:
                span = (0.0, 1.0)
            else:
                span = (float(data_finite.min()), float(data_finite.max()))
        else:
            span = (float(vmin), float(vmax))
        if span[0] >= span[1]:
            span = (span[0], span[0] + 1e-10)

        img = tf.shade(agg, cmap=cmap_obj, how="linear", span=span)
        pil_img = img.to_pil()

        # Composite onto white background to drop the alpha channel; cuts
        # PNG size by ~25% without visible difference for opaque heatmaps.
        if pil_img.mode == "RGBA":
            rgb_img = Image.new("RGB", pil_img.size, (255, 255, 255))
            rgb_img.paste(pil_img, mask=pil_img.split()[3])
            return rgb_img
        return pil_img

    def _get_colormap(self, cmap: str) -> list[str]:
        """Resolve a matplotlib colormap name to a Datashader hex-color list."""
        import matplotlib
        import matplotlib.colors as mcolors

        try:
            mpl_cmap = matplotlib.colormaps.get_cmap(cmap)
        except (ValueError, KeyError):
            logger.warning("Colormap %r not found; falling back to 'jet'", cmap)
            mpl_cmap = matplotlib.colormaps.get_cmap("jet")

        # Sample 256 colors and convert to hex — Datashader's preferred form.
        colors = [mpl_cmap(i) for i in np.linspace(0, 1, 256)]
        return [mcolors.rgb2hex(c[:3]) for c in colors]


def plot_c2_heatmap_fast(
    c2_data: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    output_path: Path,
    title: str = "",
    phi_angle: float | None = None,
    cmap: str = "jet",
    width: int = 800,
    height: int = 800,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    adaptive: bool = False,
    percentile_min: float = 1.0,
    percentile_max: float = 99.0,
) -> None:
    """Single-panel C2 heatmap via the Datashader hybrid pipeline.

    Rasterizes ``c2_data`` with Datashader (fast CPU path), displays the
    pre-rasterized image in a matplotlib figure (cheap), adds a colorbar
    using the original data range, then saves.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    renderer = DatashaderRenderer(width=width, height=height)

    # Use only finite values for adaptive percentile limits so that all-NaN
    # arrays don't produce NaN color limits (nanpercentile returns NaN on all-NaN
    # input, which bypasses the 1.0/1.5 defaults because auto_v* is not None).
    c2_finite = c2_data[np.isfinite(c2_data)]
    auto_vmin = vmin
    auto_vmax = vmax
    if adaptive and c2_finite.size > 0:
        if vmin is None:
            auto_vmin = float(np.percentile(c2_finite, percentile_min))
        if vmax is None:
            auto_vmax = float(np.percentile(c2_finite, percentile_max))
    vmin_use = auto_vmin if auto_vmin is not None else 1.0
    vmax_use = auto_vmax if auto_vmax is not None else 1.5

    # c2[t1_idx, t2_idx] → c2.T so display axes match x=t1, y=t2.
    img_pil = renderer.rasterize_heatmap(c2_data.T, t1, t2, cmap=cmap, vmin=vmin_use, vmax=vmax_use)
    img_array = np.array(img_pil)
    # Datashader y=0 at top, matplotlib origin='lower' y=0 at bottom.
    img_array = np.flipud(img_array)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)
    try:
        extent = (float(t1[0]), float(t1[-1]), float(t2[0]), float(t2[-1]))
        ax.imshow(img_array, extent=extent, origin="lower", aspect="equal")
        ax.set_xlabel("t₁ (s)", fontsize=11)
        ax.set_ylabel("t₂ (s)", fontsize=11)

        if phi_angle is not None:
            title = f"{title} at φ={phi_angle:.1f}°" if title else f"φ={phi_angle:.1f}°"
        ax.set_title(title, fontsize=13, fontweight="bold")

        norm = Normalize(vmin=vmin_use, vmax=vmax_use)
        sm = ScalarMappable(cmap=matplotlib.colormaps.get_cmap(cmap), norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label="g₂(t₁,t₂)", shrink=0.9)
        cbar.ax.tick_params(labelsize=9)

        fig.tight_layout()
        # Validate the save path (traversal / extension / null-byte) before writing.
        validated_path = validate_plot_save_path(output_path, require_parent_exists=False)
        fig.savefig(validated_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    logger.debug("Saved Datashader plot: %s", Path(output_path).name)


def plot_c2_comparison_fast(
    c2_exp: np.ndarray,
    c2_fit: np.ndarray,
    residuals: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    output_path: Path,
    phi_angle: float,
    width: int = 800,
    height: int = 800,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    adaptive: bool = True,
    percentile_min: float = 1.0,
    percentile_max: float = 99.0,
) -> None:
    """Three-panel comparison (Experimental | Fitted | Residuals) via Datashader.

    Pipeline per panel:
      1. Datashader ``rasterize_heatmap`` produces a PIL Image at the target
         resolution (default 800x800) directly from raw c2 arrays.
      2. PIL → numpy → ``np.flipud`` to convert from image coords (y=0 top)
         to math coords (y=0 bottom, matching ``origin="lower"``).
      3. matplotlib ``imshow`` displays the pre-rasterized image inside a
         figure that holds the colorbar, titles, and axis ticks.

    Exp + Fit share a color scale (computed adaptively from the **combined**
    finite-percentile range of both arrays). Residuals use their own
    data-driven range to highlight structure.

    On a 10000×10000 c2 grid: matplotlib-only takes seconds per panel and
    quickly runs out of memory; this hybrid pipeline runs in ~60-150 ms.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    renderer = DatashaderRenderer(width=width, height=height)

    # Shared color limits from combined finite values — prevents NaN limits when
    # one or both arrays are all-NaN (nanpercentile on all-NaN returns NaN, which
    # bypasses the 1.0/1.5 defaults because the variable is no longer None).
    combined_finite = np.concatenate(
        [
            c2_exp[np.isfinite(c2_exp)],
            c2_fit[np.isfinite(c2_fit)],
        ]
    )
    vmin_shared = vmin
    vmax_shared = vmax
    if adaptive and combined_finite.size > 0:
        if vmin_shared is None:
            vmin_shared = float(np.percentile(combined_finite, percentile_min))
        if vmax_shared is None:
            vmax_shared = float(np.percentile(combined_finite, percentile_max))
    vmin_shared = 1.0 if vmin_shared is None else vmin_shared
    vmax_shared = 1.5 if vmax_shared is None else vmax_shared

    img_exp = renderer.rasterize_heatmap(
        c2_exp.T, t1, t2, cmap="jet", vmin=vmin_shared, vmax=vmax_shared
    )
    img_fit = renderer.rasterize_heatmap(
        c2_fit.T, t1, t2, cmap="jet", vmin=vmin_shared, vmax=vmax_shared
    )

    # Residual colormap: symmetric ±99th-percentile of |residuals| so that
    # RdBu_r midpoint (white) always maps to zero — consistent with the
    # matplotlib path in plot_nlsq_fit. The original code used data_min/data_max
    # (asymmetric) AND discarded finite_r in favour of nanmin(residuals) which
    # could include inf values.
    finite_r = residuals[np.isfinite(residuals)] if residuals.size > 0 else residuals
    if finite_r.size > 0:
        vmax_r = float(np.percentile(np.abs(finite_r), 99))
        if vmax_r == 0.0 or not np.isfinite(vmax_r):
            vmax_r = 1.0
    else:
        vmax_r = 1.0
    res_min, res_max = -vmax_r, vmax_r
    img_res = renderer.rasterize_heatmap(
        residuals.T, t1, t2, cmap="RdBu_r", vmin=res_min, vmax=res_max
    )

    img_exp_arr = np.flipud(np.array(img_exp))
    img_fit_arr = np.flipud(np.array(img_fit))
    img_res_arr = np.flipud(np.array(img_res))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    try:
        extent = (float(t1[0]), float(t1[-1]), float(t2[0]), float(t2[-1]))

        axes[0].imshow(img_exp_arr, extent=extent, origin="lower", aspect="equal")
        axes[0].set_title(f"Experimental C₂ (φ={phi_angle:.1f}°)", fontsize=12)
        axes[0].set_xlabel("t₁ (s)", fontsize=10)
        axes[0].set_ylabel("t₂ (s)", fontsize=10)
        norm_shared = Normalize(vmin=vmin_shared, vmax=vmax_shared)
        sm_exp = ScalarMappable(cmap=matplotlib.colormaps.get_cmap("jet"), norm=norm_shared)
        sm_exp.set_array([])
        fig.colorbar(sm_exp, ax=axes[0], label="C₂(t₁,t₂)").ax.tick_params(labelsize=8)

        axes[1].imshow(img_fit_arr, extent=extent, origin="lower", aspect="equal")
        axes[1].set_title(f"Fitted C₂ (φ={phi_angle:.1f}°)", fontsize=12)
        axes[1].set_xlabel("t₁ (s)", fontsize=10)
        axes[1].set_ylabel("t₂ (s)", fontsize=10)
        sm_fit = ScalarMappable(cmap=matplotlib.colormaps.get_cmap("jet"), norm=norm_shared)
        sm_fit.set_array([])
        fig.colorbar(sm_fit, ax=axes[1], label="C₂(t₁,t₂)").ax.tick_params(labelsize=8)

        axes[2].imshow(img_res_arr, extent=extent, origin="lower", aspect="equal")
        axes[2].set_title(f"Residuals (φ={phi_angle:.1f}°)", fontsize=12)
        axes[2].set_xlabel("t₁ (s)", fontsize=10)
        axes[2].set_ylabel("t₂ (s)", fontsize=10)
        norm_res = Normalize(vmin=res_min, vmax=res_max)
        sm_res = ScalarMappable(cmap=matplotlib.colormaps.get_cmap("RdBu_r"), norm=norm_res)
        sm_res.set_array([])
        fig.colorbar(sm_res, ax=axes[2], label="ΔC₂").ax.tick_params(labelsize=8)

        fig.tight_layout()
        # Validate the save path (traversal / extension / null-byte) before writing.
        validated_path = validate_plot_save_path(output_path, require_parent_exists=False)
        fig.savefig(validated_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    logger.debug("Saved Datashader 3-panel plot: %s", Path(output_path).name)
