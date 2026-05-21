"""NLSQ fit visualization and artifact serialization.

Symbols defined here are wired into ``xpcsjax.viz``'s lazy export map by later
tasks (Task 2 onward).
"""

from __future__ import annotations

import io
import json
import multiprocessing
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from xpcsjax.io.json_utils import json_serializer
from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = get_logger(__name__)


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


def _save_fig(fig: Figure, save_path: Path | str | None, dpi: int = 150) -> None:
    """Save figure to disk and close. No-op when ``save_path`` is None.

    Creates parent directories as needed. Logs the saved path at INFO level.
    """
    if save_path is None:
        return
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", p)


def _unpack_result_params(
    model: Any,
    result: Any,
    config: dict[str, Any],
) -> tuple[float, float, np.ndarray, list[str]]:
    """Extract ``(contrast, offset, physical_params, names)`` per model type.

    HomodyneModel
        ``result.parameters[0]`` is contrast, ``[1]`` is offset, ``[2:]`` are the
        physical params. ``parameter_names`` excludes contrast/offset.

    HeterodyneModel
        ``contrast`` and ``offset`` are named slots inside the 14-element registry
        vector. ``physical_params`` is the full 14-element vector (the
        ``compute_g1`` API consumes the whole vector). ``parameter_names`` is the
        full 14-element registry-ordered name list.
    """
    from xpcsjax.core.heterodyne_model import HeterodyneModel
    from xpcsjax.core.homodyne_model import HomodyneModel

    if isinstance(model, HomodyneModel):
        params = np.asarray(result.parameters, dtype=float)
        if params.size < 3:
            raise ValueError(
                f"HomodyneModel needs >=3 params (contrast, offset, physical...); got {params.size}"
            )
        # Resolve names from either the wrapper or its inner CombinedModel; no
        # hardcoded fallback — if upstream refactors so neither attribute exists,
        # the AttributeError surfaces the real bug instead of silently lying.
        names_obj = getattr(model, "parameter_names", None)
        if names_obj is None:
            inner = getattr(model, "model", None)
            names_obj = getattr(inner, "parameter_names", None)
        if names_obj is None:
            raise AttributeError(
                "HomodyneModel exposes no parameter_names (neither directly "
                "nor via .model). xpcsjax viz cannot label physical parameters."
            )
        full_names = list(names_obj)
        physical_params = params[2:].copy()
        # Slice names to match the actual physical-param count. In static mode
        # the inner CombinedModel has 3 names and physical_params is length 3;
        # in laminar_flow it has 7 names and physical_params is length 7 — so
        # the slice is a no-op in valid cases. The slice guards against a
        # length mismatch silently corrupting downstream labels.
        names = full_names[: physical_params.size]
        return float(params[0]), float(params[1]), physical_params, names

    if isinstance(model, HeterodyneModel):
        params = np.asarray(result.parameters, dtype=float)
        names = list(model.parameter_names)
        if params.size != len(names):
            raise ValueError(f"HeterodyneModel expects {len(names)} params; got {params.size}")
        if "contrast" in names and "offset" in names:
            c = float(params[names.index("contrast")])
            o = float(params[names.index("offset")])
        else:
            raise ValueError(
                "HeterodyneModel parameter_names registry is missing required "
                "'contrast' and/or 'offset' slots."
            )
        return c, o, params.copy(), names

    raise TypeError(
        f"Unsupported model type: {type(model).__name__}. "
        f"Expected HomodyneModel or HeterodyneModel."
    )


def _evaluate_c2_per_angle(
    model: Any,
    result: Any,
    data: dict[str, Any],
    config: dict[str, Any],
    phi_deg: float,
) -> np.ndarray:
    """Compute fitted c2 surface at one phi angle.

    Dispatches on model type:

    HomodyneModel
        Uses ``_unpack_result_params`` to extract contrast/offset/physical_params,
        then calls ``model.compute_c2_single_angle(physical_params, phi, contrast,
        offset)`` which uses the model's stored t-grid/q/L/dt state.

    HeterodyneModel
        Not yet wired up. ``HeterodyneModel.compute_g1`` returns g1² (range
        [0, 1]), not a fittable c2 surface. The real c2 reconstruction needs
        per-angle contrast/offset from
        ``xpcsjax.optimization.nlsq.heterodyne_scaling_utils`` whose formulas
        vary by analysis mode (constant/auto/fourier/individual). Out of scope
        for Task 5; raises ``NotImplementedError`` until a follow-up task
        wires it up. See plan spec amendment 3.
    """
    from xpcsjax.core.heterodyne_model import HeterodyneModel
    from xpcsjax.core.homodyne_model import HomodyneModel

    if isinstance(model, HomodyneModel):
        contrast, offset, physical_params, _ = _unpack_result_params(model, result, config)
        c2 = model.compute_c2_single_angle(physical_params, phi_deg, contrast, offset)
        return np.asarray(c2)

    if isinstance(model, HeterodyneModel):
        raise NotImplementedError(
            "Heterodyne c2 reconstruction in viz is not yet wired up. "
            "HeterodyneModel.compute_g1 returns g1² (range [0, 1]), not a "
            "fittable c2 surface — the real c2 needs per-angle contrast/offset "
            "from xpcsjax.optimization.nlsq.heterodyne_scaling_utils, with "
            "formulas that vary by analysis mode (constant/auto/fourier/individual). "
            "See plan spec amendment 3."
        )

    raise TypeError(
        f"Unsupported model type: {type(model).__name__}. "
        f"Expected HomodyneModel or HeterodyneModel."
    )


def plot_nlsq_fit(
    c2_exp: np.ndarray,
    c2_fit: np.ndarray,
    t: np.ndarray | None = None,
    phi_deg: float | None = None,
    reduced_chi_squared: float | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (15, 5),
) -> Figure:
    """Three-panel NLSQ fit comparison: Experimental | Fitted | Residuals.

    Exp + Fit panels share a color scale clamped to ``[max(1.0, data_min),
    min(1.5, data_max)]`` over the **union** of both arrays so amplitude
    mismatch is visually obvious. The residual panel uses ``RdBu_r`` with
    symmetric ``±99th-percentile-of-|residual|`` limits.

    Parameters
    ----------
    c2_exp, c2_fit
        Experimental and fitted correlation surfaces, shape ``(n_t1, n_t2)``.
    t
        Optional time axis (seconds). If ``None``, uses index axes ``[0, n_t1-1]``.
    phi_deg
        Optional phi angle for per-panel titles.
    reduced_chi_squared
        If provided, appears in the super-title as ``χ²_red = {val:.3f}``.
    save_path
        If provided, the figure is saved and closed. Otherwise the live Figure is
        returned.
    figsize
        Matplotlib figsize in inches.

    Returns
    -------
    Figure
        The matplotlib Figure (open if ``save_path`` is None, closed otherwise).
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    if c2_exp.size == 0 or c2_fit.size == 0:
        fig.suptitle("No data available")
        if save_path is not None:
            _save_fig(fig, save_path)
        return fig

    n_t1, _ = c2_exp.shape
    t_arr = np.asarray(t) if t is not None else np.arange(n_t1, dtype=float)
    extent = (float(t_arr[0]), float(t_arr[-1]), float(t_arr[0]), float(t_arr[-1]))

    combined = np.concatenate([c2_exp.ravel(), c2_fit.ravel()])
    finite = combined[np.isfinite(combined)]
    data_min = float(np.nanmin(finite)) if finite.size > 0 else 1.0
    data_max = float(np.nanmax(finite)) if finite.size > 0 else 1.5
    vmin_shared = max(1.0, data_min)
    vmax_shared = min(1.5, data_max)
    if vmin_shared >= vmax_shared:
        vmax_shared = vmin_shared + 0.5

    phi_str = f" (φ={phi_deg:.1f}°)" if phi_deg is not None else ""

    im0 = axes[0].imshow(
        c2_exp,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="jet",
        vmin=vmin_shared,
        vmax=vmax_shared,
    )
    axes[0].set_title(f"Experimental Data{phi_str}")
    axes[0].set_xlabel("t₂")
    axes[0].set_ylabel("t₁")
    plt.colorbar(im0, ax=axes[0], label="c₂")

    im1 = axes[1].imshow(
        c2_fit,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="jet",
        vmin=vmin_shared,
        vmax=vmax_shared,
    )
    axes[1].set_title(f"Fitted Model{phi_str}")
    axes[1].set_xlabel("t₂")
    axes[1].set_ylabel("t₁")
    plt.colorbar(im1, ax=axes[1], label="c₂")

    residual = c2_exp - c2_fit
    finite_r = residual[np.isfinite(residual)]
    vmax_r = float(np.nanpercentile(np.abs(finite_r), 99)) if finite_r.size > 0 else 1.0
    if vmax_r == 0.0 or not np.isfinite(vmax_r):
        vmax_r = 1.0
    im2 = axes[2].imshow(
        residual,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax_r,
        vmax=vmax_r,
    )
    axes[2].set_title(f"Residuals{phi_str}")
    axes[2].set_xlabel("t₂")
    axes[2].set_ylabel("t₁")
    plt.colorbar(im2, ax=axes[2], label="Residual")

    if reduced_chi_squared is not None:
        fig.suptitle(
            f"NLSQ Fit Results  χ²_red = {reduced_chi_squared:.3f}",
            fontsize=12,
            fontweight="bold",
        )

    fig.tight_layout()

    if save_path is not None:
        _save_fig(fig, save_path)

    return fig


def plot_residual_map(
    c2_exp: np.ndarray,
    c2_fit: np.ndarray,
    t: np.ndarray | None = None,
    phi_deg: float | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (10, 10),
) -> Figure:
    """Four-panel residual diagnostic.

    Layout (2x2):
        [0,0] Residual Map (RdBu_r heatmap)
        [0,1] Residual Distribution (histogram + Normal overlay)
        [1,0] Diagonal Residuals (line trace along t1 = t2)
        [1,1] Residuals vs Fitted (scatter)

    Parameters
    ----------
    c2_exp, c2_fit
        Experimental and fitted correlation surfaces, shape ``(n_t1, n_t2)``.
    t
        Optional time axis. Falls back to index axis when None.
    phi_deg
        Optional phi for super-title.
    save_path
        If provided, saved and closed; else returned.
    figsize
        Matplotlib figsize in inches.

    Returns
    -------
    Figure
        The matplotlib Figure (open if save_path is None, closed otherwise).
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    if c2_exp.size == 0 or c2_fit.size == 0:
        fig.suptitle("No data available")
        if save_path is not None:
            _save_fig(fig, save_path)
        return fig

    residuals = c2_exp - c2_fit
    n_t = residuals.shape[0]
    t_arr = np.asarray(t) if t is not None else np.arange(n_t, dtype=float)
    extent = (float(t_arr[0]), float(t_arr[-1]), float(t_arr[0]), float(t_arr[-1]))

    # [0,0] Residual Map
    finite_r = residuals[np.isfinite(residuals)]
    vmax = float(np.nanpercentile(np.abs(finite_r), 99)) if finite_r.size > 0 else 1.0
    if vmax == 0.0 or not np.isfinite(vmax):
        vmax = 1.0
    im = axes[0, 0].imshow(
        residuals,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    axes[0, 0].set_title("Residual Map")
    axes[0, 0].set_xlabel("t₂")
    axes[0, 0].set_ylabel("t₁")
    plt.colorbar(im, ax=axes[0, 0])

    # [0,1] Histogram + Normal overlay
    flat_finite = residuals.ravel()[np.isfinite(residuals.ravel())]
    if flat_finite.size > 0:
        axes[0, 1].hist(flat_finite, bins=50, density=True, alpha=0.7)
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "No finite residuals",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
        )
    axes[0, 1].set_xlabel("Residual Value")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Residual Distribution")
    mu = float(np.nanmean(residuals)) if flat_finite.size > 0 else 0.0
    sigma = float(np.nanstd(residuals)) if flat_finite.size > 0 else 0.0
    if np.isfinite(sigma) and sigma > 0:
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        pdf = np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
        axes[0, 1].plot(
            x,
            pdf,
            "r-",
            lw=2,
            label=f"Normal(μ={mu:.2e}, σ={sigma:.2e})",
        )
        axes[0, 1].legend()

    # [1,0] Diagonal residuals
    diag = np.diag(residuals)
    axes[1, 0].plot(t_arr, diag, "b-", lw=1)
    axes[1, 0].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Residual")
    axes[1, 0].set_title("Diagonal Residuals")

    # [1,1] Residuals vs Fitted
    axes[1, 1].scatter(c2_fit.ravel(), residuals.ravel(), alpha=0.1, s=1)
    axes[1, 1].axhline(0, color="r", linestyle="--")
    axes[1, 1].set_xlabel("Fitted Value")
    axes[1, 1].set_ylabel("Residual")
    axes[1, 1].set_title("Residuals vs Fitted")

    if phi_deg is not None:
        fig.suptitle(
            f"NLSQ Residual Diagnostics  (φ={phi_deg:.1f}°)",
            fontsize=12,
            fontweight="bold",
        )

    fig.tight_layout()
    if save_path is not None:
        _save_fig(fig, save_path)
    return fig


def plot_simulated_data(
    c2_sim: np.ndarray,
    t: np.ndarray | None = None,
    phi_deg: float | None = None,
    contrast: float | None = None,
    offset: float | None = None,
    analysis_mode: str | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (8, 7),
) -> Figure:
    """Single-panel theoretical/fitted c2 heatmap with inline stats annotation.

    Used by the orchestrator to render fitted-only simulations (no comparison
    to experimental data). Annotates mean, range, and optional fit metadata
    (analysis_mode, contrast, offset).

    Parameters
    ----------
    c2_sim
        Theoretical or fitted c2 surface, shape ``(n_t1, n_t2)``.
    t
        Optional time axis.
    phi_deg
        Optional phi angle for title.
    contrast, offset, analysis_mode
        Optional metadata annotations rendered in a corner box.
    save_path
        If provided, saved and closed; else returned.
    figsize
        Matplotlib figsize in inches.

    Returns
    -------
    Figure
        The matplotlib Figure (open if save_path is None, closed otherwise).
    """
    fig, ax = plt.subplots(figsize=figsize)
    n_t1, _ = c2_sim.shape
    t_arr = np.asarray(t) if t is not None else np.arange(n_t1, dtype=float)
    extent = (float(t_arr[0]), float(t_arr[-1]), float(t_arr[0]), float(t_arr[-1]))

    vmin, vmax = _resolve_color_limits(c2_sim, percentile_min=1.0, percentile_max=99.0)
    vmin = max(1.0, vmin)
    vmax = min(1.6, vmax) if vmax > 1.0 else vmax

    im = ax.imshow(
        c2_sim.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="jet",
        interpolation="bilinear",
        vmin=vmin,
        vmax=vmax,
    )
    title = "Simulated C₂(t₁, t₂)"
    if phi_deg is not None:
        title = f"{title} at φ={phi_deg:.1f}°"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("t₁ (s)" if t is not None else "t₁ Index", fontsize=11)
    ax.set_ylabel("t₂ (s)" if t is not None else "t₂ Index", fontsize=11)
    cbar = plt.colorbar(im, ax=ax, label="C₂", shrink=0.9)
    cbar.ax.tick_params(labelsize=9)

    finite = c2_sim[np.isfinite(c2_sim)]
    if finite.size > 0:
        mean_v = float(np.nanmean(c2_sim))
        min_v = float(np.nanmin(c2_sim))
        max_v = float(np.nanmax(c2_sim))
        ax.text(
            0.02,
            0.98,
            f"Mean: {mean_v:.4f}\nRange: [{min_v:.4f}, {max_v:.4f}]",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    meta_lines = []
    if analysis_mode is not None:
        meta_lines.append(f"Mode: {analysis_mode}")
    if contrast is not None:
        meta_lines.append(f"Contrast: {contrast:.3f}")
    if offset is not None:
        meta_lines.append(f"Offset: {offset:.3f}")
    if meta_lines:
        ax.text(
            0.02,
            0.02,
            "\n".join(meta_lines),
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            bbox={"boxstyle": "round", "facecolor": "lightgreen", "alpha": 0.7},
        )

    fig.tight_layout()
    if save_path is not None:
        _save_fig(fig, save_path)
    return fig


_COMPRESSION_MAP = {
    "lzma": zipfile.ZIP_LZMA,
    "deflate": zipfile.ZIP_DEFLATED,
    "none": zipfile.ZIP_STORED,
}


def _write_npz_compressed(
    path: Path,
    arrays: dict[str, np.ndarray],
    *,
    compression: Literal["lzma", "deflate", "none"] = "lzma",
) -> None:
    """Write numerical arrays to .npz with configurable compression.

    Atomic rename: writes to ``path.tmp`` then renames. Cleans up the temp file
    on any failure.

    Compression options:
    - ``"lzma"``: best ratio, slow encode (~5-10x DEFLATE).
    - ``"deflate"``: level 9, fast and reasonable ratio.
    - ``"none"``: store only, no compression.

    ``np.load`` reads any of these transparently because the .npz container
    is just a zipfile of .npy entries; the compression method is per-entry.

    Note: arrays must be numerical only (no object-dtype). String metadata
    belongs in the JSON sidecar -- see Task 10.
    """
    if compression not in _COMPRESSION_MAP:
        raise ValueError(f"compression must be one of {set(_COMPRESSION_MAP)}; got {compression!r}")
    method = _COMPRESSION_MAP[compression]
    compresslevel = 9 if compression == "deflate" else None

    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with zipfile.ZipFile(
            tmp_path,
            mode="w",
            compression=method,
            compresslevel=compresslevel,
            allowZip64=True,
        ) as zf:
            for name, arr in arrays.items():
                arr_np = np.asarray(arr)
                if arr_np.dtype == object:
                    raise TypeError(
                        f"array {name!r} has object dtype; NPZ requires numerical "
                        "arrays only (string metadata belongs in the JSON sidecar)"
                    )
                buf = io.BytesIO()
                np.lib.format.write_array(buf, arr_np)
                zf.writestr(f"{name}.npy", buf.getvalue())
        tmp_path.replace(path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _save_fit_artifacts(
    *,
    c2_exp: np.ndarray,
    c2_fitted: np.ndarray,
    residuals: np.ndarray,
    phi_angles: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    q: float,
    L: float,
    dt: float,
    params: np.ndarray,
    uncertainties: np.ndarray,
    parameter_names: list[str],
    contrast: float,
    offset: float,
    reduced_chi_squared: float,
    convergence_status: str,
    iterations: int,
    execution_time: float,
    analysis_mode: str,
    output_dir: Path,
    compression: Literal["lzma", "deflate", "none"] = "lzma",
) -> None:
    """Serialize fitted artifacts: NPZ (numerical) + JSON (metadata + strings).

    LZMA OSError/MemoryError automatically falls back to DEFLATE-9 with a
    logged warning. JSON is written atomically (tmp + rename) to mirror the
    NPZ guarantee that mid-write failures leave no stale files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "c2_fitted_data.npz"
    json_path = output_dir / "simulation_config_fitted.json"

    arrays = {
        "c2_exp": c2_exp,
        "c2_fitted": c2_fitted,
        "residuals": residuals,
        "phi_angles": phi_angles,
        "t1": t1,
        "t2": t2,
        "q": np.float64(q),
        "params": params,
        "contrast": np.float64(contrast),
        "offset": np.float64(offset),
        "reduced_chi_squared": np.float64(reduced_chi_squared),
    }

    try:
        _write_npz_compressed(npz_path, arrays, compression=compression)
    except (OSError, MemoryError) as e:
        if compression == "lzma":
            logger.warning("LZMA compression failed (%s); falling back to DEFLATE-9", e)
            _write_npz_compressed(npz_path, arrays, compression="deflate")
        else:
            raise

    meta = {
        "fit": {
            "parameters": {
                "values": [float(v) for v in np.asarray(params).ravel()],
                "uncertainties": [float(v) for v in np.asarray(uncertainties).ravel()],
                "names": list(parameter_names),
            },
            "contrast": float(contrast),
            "offset": float(offset),
            "reduced_chi_squared": float(reduced_chi_squared),
            "convergence_status": str(convergence_status),
            "iterations": int(iterations),
            "execution_time": float(execution_time),
        },
        "physics": {
            "q_value_angstrom_inv": float(q),
            "stator_rotor_gap_angstrom": float(L),
            "dt": float(dt),
            "analysis_mode": str(analysis_mode),
        },
        "data": {
            "n_phi": int(phi_angles.shape[0]),
            "n_t1": int(t1.shape[0]),
            "n_t2": int(t2.shape[0]),
            "phi_angles_deg": [float(p) for p in np.asarray(phi_angles).ravel()],
        },
    }

    tmp_json = json_path.with_suffix(json_path.suffix + ".tmp")
    try:
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=json_serializer)
        tmp_json.replace(json_path)
    except BaseException:
        if tmp_json.exists():
            tmp_json.unlink(missing_ok=True)
        raise

    logger.info("Wrote fit artifacts to %s", output_dir)


def _worker_init_cpu_only() -> None:
    """Pool worker initializer — pin JAX to CPU + lazy allocator."""
    import os

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def _render_one_angle_worker(args: tuple) -> None:
    """Picklable worker: receives arrays + paths, writes PNGs.

    Re-imports plot funcs inside the worker (spawn-context starts cold).
    Used in both the parallel path (executed in a Pool worker process) and
    the sequential / fallback path (executed in the main process) — so output
    is byte-identical regardless of which path produced it.
    """
    import os

    os.environ["JAX_PLATFORMS"] = "cpu"

    from xpcsjax.viz.nlsq_plots import (
        plot_nlsq_fit,
        plot_residual_map,
        plot_simulated_data,
    )

    (
        c2_exp_i,
        c2_fit_i,
        t1,
        phi_deg,
        plots,
        chi2_red,
        contrast,
        offset,
        analysis_mode,
        output_dir,
        sim_dir,
    ) = args

    if "comparison" in plots:
        plot_nlsq_fit(
            c2_exp_i,
            c2_fit_i,
            t=t1,
            phi_deg=phi_deg,
            reduced_chi_squared=chi2_red,
            save_path=Path(output_dir) / f"c2_heatmaps_phi_{phi_deg:.1f}deg.png",
        )
    if "residuals" in plots:
        plot_residual_map(
            c2_exp_i,
            c2_fit_i,
            t=t1,
            phi_deg=phi_deg,
            save_path=Path(output_dir) / f"residuals_phi_{phi_deg:.1f}deg.png",
        )
    if "simulated" in plots:
        plot_simulated_data(
            c2_fit_i,
            t=t1,
            phi_deg=phi_deg,
            contrast=contrast,
            offset=offset,
            analysis_mode=analysis_mode,
            save_path=Path(sim_dir) / f"simulated_c2_fitted_phi_{phi_deg:.1f}deg.png",
        )


def generate_nlsq_plots(
    model: Any,
    result: Any,
    data: dict[str, Any],
    config: Any,
    output_dir: Path | str,
    *,
    use_datashader: bool = False,
    parallel: bool = False,
    plots: tuple[str, ...] = ("comparison", "residuals", "simulated"),
    compression: Literal["lzma", "deflate", "none"] = "lzma",
) -> None:
    """Generate NLSQ fit plots and serialize fitted artifacts.

    For each phi angle: recompute the fitted c2 surface via model dispatch,
    write PNG files for the selected plot families, then dump NPZ + JSON
    artifacts under ``output_dir/simulated_data/``.

    Parameters
    ----------
    model
        Currently only ``HomodyneModel`` is supported. ``HeterodyneModel``
        raises ``NotImplementedError`` — see Spec Amendment 3 (heterodyne
        c2 reconstruction needs ``heterodyne_scaling_utils`` integration
        that's pending).
    result
        ``OptimizationResult`` from ``fit_nlsq``.
    data
        Dict with keys: ``c2_exp`` (n_phi, n_t1, n_t2), ``phi_angles_list``,
        ``t1``, ``t2``.
    config
        ``ConfigManager`` instance or dict. Must contain
        ``analyzer_parameters.scattering.wavevector_q``,
        ``analyzer_parameters.geometry.stator_rotor_gap``,
        ``analyzer_parameters.dt``, and ``analysis_mode``.
    output_dir
        Directory to write into. Created if missing. ``simulated_data/``
        subdirectory is also created.
    use_datashader
        If True and datashader is installed, use the fast preview backend
        (currently logs a warning and falls back to matplotlib — datashader
        integration lands in Task 16).
    parallel
        If True, render angles in a ``multiprocessing.Pool(spawn)`` — wired
        in Task 14. Defaults to False (sequential).
    plots
        Subset of ``{"comparison", "residuals", "simulated"}``.
    compression
        NPZ compression: ``"lzma"`` (default, best ratio), ``"deflate"``,
        or ``"none"``.

    Raises
    ------
    ValueError
        Unknown plot family, invalid compression, missing required data key,
        shape mismatch, or missing physics keys in config.
    TypeError
        Unsupported model type (not HomodyneModel or HeterodyneModel).
    NotImplementedError
        HeterodyneModel is currently deferred (Spec Amendment 3).
    """
    from xpcsjax.core.heterodyne_model import HeterodyneModel
    from xpcsjax.core.homodyne_model import HomodyneModel

    # Fail-fast on heterodyne — per code review recommendation, isinstance
    # check upstream is cleaner than catching NotImplementedError from
    # _evaluate_c2_per_angle.
    if isinstance(model, HeterodyneModel):
        raise NotImplementedError(
            "generate_nlsq_plots does not yet support HeterodyneModel — "
            "heterodyne c2 reconstruction needs heterodyne_scaling_utils "
            "integration (Spec Amendment 3). Use the homodyne path for now."
        )
    if not isinstance(model, HomodyneModel):
        raise TypeError(
            f"Unsupported model type: {type(model).__name__}. "
            f"Expected HomodyneModel or HeterodyneModel."
        )

    # Resolve config to a plain dict
    config_dict = config.config if hasattr(config, "config") else config

    # Validation
    valid = {"comparison", "residuals", "simulated"}
    unknown = set(plots) - valid
    if unknown:
        raise ValueError(f"Unknown plot families: {sorted(unknown)}. Valid: {sorted(valid)}")
    if compression not in {"lzma", "deflate", "none"}:
        raise ValueError(f"compression must be 'lzma', 'deflate', or 'none'; got {compression!r}")

    for key in ("c2_exp", "phi_angles_list", "t1", "t2"):
        if key not in data:
            raise ValueError(f"data dict missing required key: {key!r}")

    c2_exp = np.asarray(data["c2_exp"])
    phi_angles = np.asarray(data["phi_angles_list"], dtype=float)
    t1 = np.asarray(data["t1"], dtype=float)
    t2 = np.asarray(data["t2"], dtype=float)
    expected_shape = (phi_angles.size, t1.size, t2.size)
    if c2_exp.shape != expected_shape:
        raise ValueError(
            f"c2_exp.shape {c2_exp.shape} does not match (n_phi, n_t1, n_t2) {expected_shape}"
        )

    ap = config_dict.get("analyzer_parameters", {})
    q = ap.get("scattering", {}).get("wavevector_q")
    L = ap.get("geometry", {}).get("stator_rotor_gap")
    dt = ap.get("dt")
    if dt is None:
        # Fall back to temporal.dt — homodyne configs nest it there.
        dt = ap.get("temporal", {}).get("dt")
    if q is None or L is None or dt is None:
        raise ValueError(
            "config.analyzer_parameters must contain scattering.wavevector_q, "
            "geometry.stator_rotor_gap, and dt (or temporal.dt)"
        )
    analysis_mode = config_dict.get("analysis_mode", "unknown")

    # Output dirs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_dir = output_dir / "simulated_data"
    sim_dir.mkdir(parents=True, exist_ok=True)

    # Per-model param unpacking (homodyne only at this point)
    contrast, offset, physical_params, parameter_names = _unpack_result_params(
        model, result, config_dict
    )

    # Pre-allocate fitted surface with NaN sentinel
    c2_fitted = np.full_like(c2_exp, np.nan, dtype=float)
    n_phi = phi_angles.size

    logger.info(
        "Generating NLSQ plots: %d angles, parallel=%s, datashader=%s, plots=%s",
        n_phi,
        parallel,
        use_datashader,
        plots,
    )

    # Datashader fallback (real integration lands in Task 16)
    if use_datashader:
        try:
            import xpcsjax.viz.datashader_backend  # noqa: F401
        except ImportError:
            logger.warning(
                "use_datashader=True but datashader is not installed. "
                "Install with: pip install 'xpcsjax[viz-fast]'. "
                "Falling back to matplotlib."
            )

    # Phase A: compute fitted surfaces in main process (models may not be picklable)
    chi2_red = float(result.reduced_chi_squared)
    for i, phi_deg in enumerate(phi_angles):
        phi_deg_f = float(phi_deg)
        try:
            c2_fitted[i] = _evaluate_c2_per_angle(model, result, data, config_dict, phi_deg_f)
        except Exception:
            logger.exception(
                "Angle %d (phi=%.1f) compute failed; leaving NaN in c2_fitted",
                i,
                phi_deg_f,
            )

    # Phase B: render PNGs (parallel pool or sequential — same worker either way
    # so the output is byte-identical regardless of path).
    def _render_args_for_index(i: int) -> tuple:
        return (
            c2_exp[i],
            c2_fitted[i],
            t1,
            float(phi_angles[i]),
            plots,
            chi2_red,
            contrast,
            offset,
            analysis_mode,
            output_dir,
            sim_dir,
        )

    if parallel and n_phi > 1:
        try:
            ctx = multiprocessing.get_context("spawn")
            n_workers = min(multiprocessing.cpu_count(), n_phi)
            args_list = [
                _render_args_for_index(i)
                for i in range(n_phi)
                if not np.all(np.isnan(c2_fitted[i]))
            ]
            timeout_s = 60 * n_phi / max(n_workers, 1) + 120
            with ctx.Pool(processes=n_workers, initializer=_worker_init_cpu_only) as pool:
                ar = pool.map_async(_render_one_angle_worker, args_list)
                ar.get(timeout=timeout_s)
            logger.info(
                "Rendered %d angles in parallel (%d workers)",
                len(args_list),
                n_workers,
            )
        except (OSError, RuntimeError, TimeoutError, Exception) as e:
            logger.warning(
                "Parallel rendering failed (%s: %s); sequential fallback.",
                type(e).__name__,
                e,
            )
            for i in range(n_phi):
                if np.all(np.isnan(c2_fitted[i])):
                    continue
                _render_one_angle_worker(_render_args_for_index(i))
    else:
        for i in range(n_phi):
            if np.all(np.isnan(c2_fitted[i])):
                continue
            _render_one_angle_worker(_render_args_for_index(i))

    residuals = c2_exp - c2_fitted

    # Slice uncertainties to match physical_params length for homodyne
    phys_unc = np.asarray(result.uncertainties, dtype=float)[2:]

    _save_fit_artifacts(
        c2_exp=c2_exp,
        c2_fitted=c2_fitted,
        residuals=residuals,
        phi_angles=phi_angles,
        t1=t1,
        t2=t2,
        q=float(q),
        L=float(L),
        dt=float(dt),
        params=np.asarray(physical_params, dtype=float),
        uncertainties=phys_unc,
        parameter_names=list(parameter_names),
        contrast=float(contrast),
        offset=float(offset),
        reduced_chi_squared=chi2_red,
        convergence_status=str(result.convergence_status),
        iterations=int(result.iterations),
        execution_time=float(result.execution_time),
        analysis_mode=str(analysis_mode),
        output_dir=sim_dir,
        compression=compression,
    )

    logger.info("NLSQ plot generation complete: %s", output_dir)
