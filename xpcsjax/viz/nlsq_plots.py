"""NLSQ fit visualization and artifact serialization.

Symbols defined here are wired into ``xpcsjax.viz``'s lazy export map by later
tasks (Task 2 onward).
"""

from __future__ import annotations

import io
import json
import multiprocessing
import os
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.io.json_utils import json_serializer
from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = get_logger(__name__)

# Optional Datashader backend probe — see xpcsjax/viz/datashader_backend.py.
# Use find_spec to verify the optional deps are present before importing, so
# that a genuine bug inside datashader_backend (SyntaxError, AttributeError,
# etc.) propagates instead of being silently swallowed as a missing-dep flag.
import importlib.util as _importlib_util

DATASHADER_AVAILABLE = False
if (
    _importlib_util.find_spec("datashader") is not None
    and _importlib_util.find_spec("xarray") is not None
):
    try:
        import xpcsjax.viz.datashader_backend  # noqa: F401

        DATASHADER_AVAILABLE = True
    except ImportError:
        # Optional deps declared missing by the backend's own guard — expected.
        DATASHADER_AVAILABLE = False


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

    Creates parent directories as needed. The figure is closed even if
    ``savefig`` raises, so renderer/filesystem errors don't leak Figure
    handles. Logs the saved path at INFO level.
    """
    if save_path is None:
        return
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    logger.info("Figure saved: %s", p)


def _is_homodyne_family(model: Any) -> bool:
    """True for models that use the homodyne result layout
    ``[contrast, offset, *physical]`` with scalar per-angle scaling.

    Two concrete types qualify: :class:`HomodyneModel` (the stateful viz
    wrapper) and the bare :class:`CombinedModel` that
    :func:`xpcsjax.core.models.make_model` returns for the homodyne analysis
    modes (``static_*`` / ``laminar_flow``; the Task-28 contract pinned by
    ``tests/config/test_get_model.py``).

    This is the single source of truth for homodyne-family membership in the
    viz layer — the acceptance gates (``_unpack_result_params`` and the
    ``generate_nlsq_plots`` guard) route through it so they cannot drift apart
    when a model type is added.
    """
    from xpcsjax.core.homodyne_model import HomodyneModel
    from xpcsjax.core.models import CombinedModel

    return isinstance(model, (HomodyneModel, CombinedModel))


def _is_heterodyne_family(model: Any) -> bool:
    """True for :class:`HeterodyneModel` (per-angle ``[c.., o.., *physical]`` layout)."""
    from xpcsjax.core.heterodyne_model import HeterodyneModel

    return isinstance(model, HeterodyneModel)


def _is_supported_viz_model(model: Any) -> bool:
    """True for any model type the viz layer knows how to plot."""
    return _is_homodyne_family(model) or _is_heterodyne_family(model)


def _homodyne_scaling_arrays(
    model: Any, result: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return per-angle ``(contrasts, offsets, physical_params, names)`` for a
    homodyne-family result.

    The homodyne NLSQ fit uses per-angle scaling by default, so
    ``result.parameters`` is laid out ``[c_0..N-1, o_0..N-1, physical_0..M-1]``
    — the *same* layout as heterodyne ``individual`` mode. The legacy scalar
    layout ``[contrast, offset, physical...]`` is simply the ``n_phi == 1`` case
    and falls out of the same slicing.

    ``n_phi`` is inferred from the model's physical-parameter count
    (``len(parameter_names)``), NOT assumed to be 1. Assuming a single
    ``[contrast, offset, ...]`` pair on a per-angle vector read ``offset`` as the
    second *contrast* (``c_1``) and shifted the physical block — which rendered
    the fitted c2 surface flat (``offset + contrast*g1²`` with ``offset`` wrongly
    ≈ ``contrast``). Inferring ``n_phi`` from the physical count fixes that.
    """
    params = np.asarray(result.parameters, dtype=float)
    # Resolve physical names from the model (CombinedModel directly; HomodyneModel
    # via its inner ``.model``). No hardcoded fallback — a missing attribute
    # surfaces the real bug instead of silently mislabeling.
    names_obj = getattr(model, "parameter_names", None)
    if names_obj is None:
        inner = getattr(model, "model", None)
        names_obj = getattr(inner, "parameter_names", None)
    if names_obj is None:
        raise AttributeError(
            "Homodyne-family model exposes no parameter_names (neither directly "
            "nor via .model); xpcsjax viz cannot determine the scaling layout."
        )
    names = list(names_obj)
    n_physical = len(names)
    n_scaling = params.size - n_physical
    if n_scaling < 0 or n_scaling % 2 != 0:
        raise ValueError(
            f"Homodyne-family result has {params.size} params but the model "
            f"declares {n_physical} physical; the scaling block ({n_scaling}) is "
            f"not a non-negative even count. Expected "
            f"[c_0..N-1, o_0..N-1, physical...] = 2*n_phi + {n_physical}."
        )
    n_phi = n_scaling // 2
    contrasts = params[:n_phi].copy()
    offsets = params[n_phi : 2 * n_phi].copy()
    physical_params = params[2 * n_phi :].copy()
    return contrasts, offsets, physical_params, names


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
    # HomodyneModel (the stateful viz wrapper) and the bare CombinedModel
    # returned by ``core.models.make_model`` for the homodyne modes
    # (static_*/laminar_flow). The homodyne NLSQ fit uses per-angle scaling by
    # default, so ``result.parameters`` is ``[c_0..N-1, o_0..N-1, *physical]`` —
    # see _homodyne_scaling_arrays. Collapse the per-angle pairs to scalars for
    # this helper's summary contract; the per-angle arrays are used directly by
    # _evaluate_c2_per_angle.
    if _is_homodyne_family(model):
        contrasts, offsets, physical_params, names = _homodyne_scaling_arrays(
            model, result
        )
        contrast_scalar = float(contrasts.mean()) if contrasts.size else 0.0
        offset_scalar = float(offsets.mean()) if offsets.size else 0.0
        return contrast_scalar, offset_scalar, physical_params, names

    if _is_heterodyne_family(model):
        params = np.asarray(result.parameters, dtype=float)
        physical_names = list(model.parameter_names)  # 14 names
        n_physical = len(physical_names)
        n_total = params.size
        diagnostics = getattr(result, "nlsq_diagnostics", None) or {}
        mode = diagnostics.get("per_angle_mode")
        # Averaged mode: [physical..., contrast, offset]. Scalar summary is the
        # single fitted pair (physics is the leading 14-vector).
        if mode == "averaged":
            physical_params = params[:n_physical].copy()
            contrast_scalar = float(
                diagnostics.get("averaged_contrast", params[n_physical])
            )
            offset_scalar = float(
                diagnostics.get("averaged_offset", params[n_physical + 1])
            )
            return contrast_scalar, offset_scalar, physical_params, physical_names
        # Constant mode: [physical...] only; scaling frozen in diagnostics.
        if mode == "constant":
            physical_params = params[:n_physical].copy()
            c_fixed = np.asarray(diagnostics["contrast_per_angle_fixed"], dtype=float)
            o_fixed = np.asarray(diagnostics["offset_per_angle_fixed"], dtype=float)
            return (
                float(c_fixed.mean()),
                float(o_fixed.mean()),
                physical_params,
                physical_names,
            )
        # Individual mode (or diagnostics-less result):
        # Per-angle layout: [c_0..N-1, o_0..N-1, physical_0..n_physical-1]
        # Require 2*n_phi + n_physical params; the residual (n_total - n_physical)
        # must be even *and* the orchestrator's upfront layout validator must
        # have already confirmed individual-mode shape — see
        # ``_assert_heterodyne_individual_layout``. We tolerate the no-scaling
        # case (n_total == n_physical) by returning zero scalars so the
        # downstream simulated-data annotation panel still renders; the real
        # heterodyne per-angle evaluation path uses
        # ``_unpack_heterodyne_scaling`` which fails loudly for that case.
        residual = n_total - n_physical
        if residual < 0 or residual % 2 != 0:
            raise ValueError(
                f"HeterodyneModel expects 2*n_phi + {n_physical} params "
                f"(per-angle layout); got {n_total}. The residual "
                f"{residual} is not divisible by 2."
            )
        n_phi = residual // 2
        contrasts = params[:n_phi]
        offsets = params[n_phi : 2 * n_phi]
        physical_params = params[2 * n_phi :].copy()
        # For the homodyne-shaped (scalar contrast, offset) return contract,
        # use the per-angle means as scalar summaries. Per-angle arrays are
        # extracted by callers that need them (see _evaluate_c2_per_angle).
        contrast_scalar = float(contrasts.mean()) if n_phi > 0 else 0.0
        offset_scalar = float(offsets.mean()) if n_phi > 0 else 0.0
        return contrast_scalar, offset_scalar, physical_params, physical_names

    raise TypeError(
        f"Unsupported model type: {type(model).__name__}. "
        f"Expected HomodyneModel, CombinedModel, or HeterodyneModel."
    )


def _unpack_heterodyne_scaling(
    model: Any,
    result: Any,
    n_phi_expected: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Extract heterodyne per-angle scaling + physical params from a result.

    The per-angle scaling mode is read from
    ``result.nlsq_diagnostics["per_angle_mode"]`` and the correct parameter
    layout is reconstructed for each:

    - ``individual`` — ``[c_0..N-1, o_0..N-1, physical_0..M-1]`` (per-angle
      contrast/offset fitted independently).
    - ``averaged`` — ``[physical..., contrast, offset]``; the single fitted
      (contrast, offset) pair is replicated across all angles.
    - ``constant`` — ``[physical...]``; per-angle scaling was frozen pre-fit
      and is read from the ``contrast_per_angle_fixed`` /
      ``offset_per_angle_fixed`` diagnostics.

    When ``nlsq_diagnostics`` is absent (e.g. a synthetic result), the layout
    is inferred from the parameter count and treated as ``individual``. The
    caller must supply ``n_phi_expected`` (from ``data["phi_angles_list"]``) so
    the individual layout can be disambiguated from the ``fourier`` mode, which
    has ``2*(2K+1)`` extra slots that would otherwise be misread as ``2*n_phi``.

    Returns
    -------
    (contrasts, offsets, physical_params, n_phi)
        contrasts, offsets: shape (n_phi_expected,)
        physical_params:    shape (n_physical=14,)
        n_phi:              equals ``n_phi_expected``

    Raises
    ------
    NotImplementedError
        ``fourier`` mode, or a diagnostics-less result whose parameter count
        matches no recognised layout. Those remain out of scope for v0.1 viz.
    """
    if not _is_heterodyne_family(model):
        raise TypeError(
            f"_unpack_heterodyne_scaling expects HeterodyneModel; got {type(model).__name__}"
        )
    params = np.asarray(result.parameters, dtype=float)
    n_physical = len(model.parameter_names)
    n_total = params.size
    diagnostics = getattr(result, "nlsq_diagnostics", None) or {}
    mode = diagnostics.get("per_angle_mode")

    # Averaged mode: layout is ``[physics..., contrast, offset]`` — one fitted
    # (contrast, offset) pair shared across all angles. Prefer the fitted
    # scalars stored in diagnostics; fall back to the trailing two parameter
    # slots. Replicate across n_phi so the per-angle evaluation path stays
    # uniform with individual mode.
    if mode == "averaged":
        physical_params = params[:n_physical].copy()
        contrast = float(diagnostics.get("averaged_contrast", params[n_physical]))
        offset = float(diagnostics.get("averaged_offset", params[n_physical + 1]))
        contrasts = np.full(n_phi_expected, contrast, dtype=float)
        offsets = np.full(n_phi_expected, offset, dtype=float)
        return contrasts, offsets, physical_params, n_phi_expected

    # Constant mode: layout is ``[physics...]`` only; per-angle scaling was
    # frozen pre-fit and is carried in diagnostics.
    if mode == "constant":
        physical_params = params[:n_physical].copy()
        contrasts = np.asarray(
            diagnostics["contrast_per_angle_fixed"], dtype=float
        ).ravel()
        offsets = np.asarray(diagnostics["offset_per_angle_fixed"], dtype=float).ravel()
        if contrasts.size != n_phi_expected or offsets.size != n_phi_expected:
            raise ValueError(
                f"Constant-mode per-angle scaling has {contrasts.size} contrasts / "
                f"{offsets.size} offsets but {n_phi_expected} angles were requested."
            )
        return contrasts, offsets, physical_params, n_phi_expected

    # Individual mode (or a diagnostics-less result, e.g. a synthetic
    # OptimizationResult): layout is ``[c_0..N-1, o_0..N-1, physics...]``.
    individual_total = n_physical + 2 * n_phi_expected
    if n_total != individual_total:
        if n_total == n_physical:
            raise NotImplementedError(
                f"Heterodyne 'constant' scaling mode is not yet supported by xpcsjax "
                f"viz. v0.1 supports per-angle 'individual' mode only — got "
                f"{n_physical} physical params with no per-angle (contrast, offset) "
                f"pairs in result.parameters. Use the upstream heterodyne package or "
                f"wait for v0.2 for full mode parity."
            )
        raise NotImplementedError(
            f"Heterodyne result has {n_total} parameters but xpcsjax viz expects "
            f"{individual_total} (individual mode: {n_physical} physics + "
            f"2*{n_phi_expected} per-angle scaling). Scaling mode "
            f"{mode!r} (e.g. 'fourier') is not yet supported by v0.1 viz."
        )
    contrasts = params[:n_phi_expected].copy()
    offsets = params[n_phi_expected : 2 * n_phi_expected].copy()
    physical_params = params[2 * n_phi_expected :].copy()
    return contrasts, offsets, physical_params, n_phi_expected


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
        Reads per-angle ``contrasts[i]`` / ``offsets[i]`` from the per-angle
        fit-time layout in ``result.parameters`` (via
        ``_unpack_heterodyne_scaling``), evaluates ``model.compute_g1`` to get
        the normalized g1² surface (range [0, 1]) for the matching angle, and
        applies ``c2 = offset[i] + contrast[i] * g1_sq``. Resolves Spec
        Amendment 3.
    """
    if _is_homodyne_family(model):
        contrasts, offsets, physical_params, _ = _homodyne_scaling_arrays(model, result)
        # Per-angle scaling: render THIS angle with its own (contrast, offset),
        # matching upstream homodyne's per-angle plots. With the scalar legacy
        # layout (n_phi == 1) the single pair applies to every angle.
        if contrasts.size <= 1:
            contrast = float(contrasts[0]) if contrasts.size else 0.0
            offset = float(offsets[0]) if offsets.size else 1.0
        else:
            phi_array = np.asarray(data["phi_angles_list"], dtype=float)
            matches = np.where(np.isclose(phi_array, phi_deg, atol=1e-6))[0]
            if matches.size == 0:
                raise ValueError(
                    f"phi_deg={phi_deg!r} not found in data['phi_angles_list'] "
                    f"(values: {phi_array.tolist()})"
                )
            i = int(matches[0])
            contrast = float(contrasts[i])
            offset = float(offsets[i])
        # HomodyneModel (the stateful wrapper) carries pre-computed grids /
        # physics-factors and exposes a single-angle helper. The bare
        # CombinedModel that ``make_model`` returns for static_*/laminar_flow
        # does not, so drive its ``compute_g2`` with q/L/dt from the config and
        # the data's time grids (``compute_g2`` applies ``offset + contrast*g1**2``
        # internally — mirrors plot_dispatch._evaluate_model_c2). Capability
        # dispatch rather than isinstance-per-type keeps any future
        # homodyne-family model working as long as it exposes one of these APIs.
        if hasattr(model, "compute_c2_single_angle"):
            c2 = model.compute_c2_single_angle(physical_params, phi_deg, contrast, offset)
            return np.asarray(c2)
        ap = config.get("analyzer_parameters", {})
        q_raw = ap.get("scattering", {}).get("wavevector_q")
        if q_raw is None:
            raise ValueError("Missing analyzer_parameters.scattering.wavevector_q")
        L_raw = ap.get("geometry", {}).get("stator_rotor_gap")
        if L_raw is None:
            raise ValueError("Missing analyzer_parameters.geometry.stator_rotor_gap")
        # Explicit None-check so dt=0 is not treated as falsy.
        dt_raw = ap.get("dt")
        if dt_raw is None:
            dt_raw = ap.get("temporal", {}).get("dt")
        if dt_raw is None:
            raise ValueError(
                "Missing analyzer_parameters: 'dt' or 'temporal.dt' is required"
            )
        t1 = jnp.asarray(data["t1"], dtype=jnp.float64)
        t2 = jnp.asarray(data["t2"], dtype=jnp.float64)
        g2 = model.compute_g2(
            jnp.asarray(physical_params, dtype=jnp.float64),
            t1,
            t2,
            jnp.asarray([phi_deg], dtype=jnp.float64),
            float(q_raw),
            float(L_raw),
            float(contrast),
            float(offset),
            float(dt_raw),
        )
        # compute_g2 returns shape (1, n_t1, n_t2) for length-1 phi; drop axis.
        return np.asarray(g2[0])

    if _is_heterodyne_family(model):
        # Locate phi_deg's index in data["phi_angles_list"] to pick the
        # right per-angle contrast/offset. Tolerance is loose since phi
        # angles are user-provided floats; exact match expected.
        phi_array = np.asarray(data["phi_angles_list"], dtype=float)
        matches = np.where(np.isclose(phi_array, phi_deg, atol=1e-6))[0]
        if matches.size == 0:
            raise ValueError(
                f"phi_deg={phi_deg!r} not found in data['phi_angles_list'] "
                f"(values: {phi_array.tolist()})"
            )
        i = int(matches[0])
        n_phi_expected = int(phi_array.size)
        contrasts, offsets, physical_params, _ = _unpack_heterodyne_scaling(
            model, result, n_phi_expected=n_phi_expected
        )

        ap = config.get("analyzer_parameters", {})
        q_raw = ap.get("scattering", {}).get("wavevector_q")
        if q_raw is None:
            raise ValueError("Missing analyzer_parameters.scattering.wavevector_q")
        q = float(q_raw)
        L_raw = ap.get("geometry", {}).get("stator_rotor_gap")
        if L_raw is None:
            raise ValueError("Missing analyzer_parameters.geometry.stator_rotor_gap")
        L = float(L_raw)
        # Use explicit None-check so dt=0 is not treated as falsy.
        dt_raw = ap.get("dt")
        if dt_raw is None:
            dt_raw = ap.get("temporal", {}).get("dt")
        if dt_raw is None:
            raise ValueError(
                "Missing analyzer_parameters: 'dt' or 'temporal.dt' is required"
            )
        dt = float(dt_raw)
        t1 = jnp.asarray(data["t1"], dtype=jnp.float64)
        t2 = jnp.asarray(data["t2"], dtype=jnp.float64)

        g1_sq = model.compute_g1(
            jnp.asarray(physical_params, dtype=jnp.float64),
            t1,
            t2,
            jnp.asarray([phi_deg], dtype=jnp.float64),
            q,
            L,
            dt,
        )
        # compute_g1 returns shape (1, n_t1, n_t2) for length-1 phi; drop axis.
        g1_sq_arr = np.asarray(g1_sq[0])
        c2 = float(offsets[i]) + float(contrasts[i]) * g1_sq_arr
        return c2

    raise TypeError(
        f"Unsupported model type: {type(model).__name__}. "
        f"Expected HomodyneModel, CombinedModel, or HeterodyneModel."
    )


def plot_nlsq_fit(
    c2_exp: np.ndarray,
    c2_fit: np.ndarray,
    t: np.ndarray | None = None,
    phi_deg: float | None = None,
    reduced_chi_squared: float | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (15, 5),
    *,
    t2: np.ndarray | None = None,
) -> Figure | None:
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
        Optional time axis (seconds) — used as the y-axis (t₁). If ``t2``
        is also ``None``, the same vector is used for both axes (square
        assumption). If ``None``, uses index axes.
    t2
        Optional x-axis (t₂). When supplied with ``t``, lets rectangular
        grids (n_t1 ≠ n_t2) render with the correct horizontal extent.
    phi_deg
        Optional phi angle for per-panel titles.
    reduced_chi_squared
        If provided, appears in the super-title as ``χ²_red = {val:.3f}``.
    save_path
        If provided, the figure is saved and closed; the function returns
        ``None``. Otherwise the live Figure is returned.
    figsize
        Matplotlib figsize in inches.

    Returns
    -------
    Figure or None
        The matplotlib Figure when ``save_path`` is ``None``; ``None`` when
        the figure was saved (and is therefore closed).
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    if c2_exp.size == 0 or c2_fit.size == 0:
        fig.suptitle("No data available")
        if save_path is not None:
            _save_fig(fig, save_path)
            return None
        return fig

    n_t1, n_t2 = c2_exp.shape
    t_y = np.asarray(t) if t is not None else np.arange(n_t1, dtype=float)
    t_x = np.asarray(t2) if t2 is not None else (t_y if t is not None else np.arange(n_t2, dtype=float))
    extent = (float(t_x[0]), float(t_x[-1]), float(t_y[0]), float(t_y[-1]))

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
        return None

    return fig


def plot_residual_map(
    c2_exp: np.ndarray,
    c2_fit: np.ndarray,
    t: np.ndarray | None = None,
    phi_deg: float | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (10, 10),
    *,
    t2: np.ndarray | None = None,
) -> Figure | None:
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
        Optional time axis (y / t₁). Falls back to index axis when None.
    t2
        Optional x-axis (t₂). When supplied with ``t``, lets rectangular
        grids (n_t1 ≠ n_t2) render with the correct horizontal extent.
    phi_deg
        Optional phi for super-title.
    save_path
        If provided, saved and closed; the function returns ``None``.
        Otherwise the live Figure is returned.
    figsize
        Matplotlib figsize in inches.

    Returns
    -------
    Figure or None
        The matplotlib Figure when ``save_path`` is ``None``; ``None`` when
        the figure was saved (and is therefore closed).
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    if c2_exp.size == 0 or c2_fit.size == 0:
        fig.suptitle("No data available")
        if save_path is not None:
            _save_fig(fig, save_path)
            return None
        return fig

    residuals = c2_exp - c2_fit
    n_t1, n_t2 = residuals.shape
    t_y = np.asarray(t) if t is not None else np.arange(n_t1, dtype=float)
    t_x = np.asarray(t2) if t2 is not None else (t_y if t is not None else np.arange(n_t2, dtype=float))
    extent = (float(t_x[0]), float(t_x[-1]), float(t_y[0]), float(t_y[-1]))

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

    # [1,0] Diagonal residuals — length is min(n_t1, n_t2); plot against t_y truncated.
    diag = np.diag(residuals)
    axes[1, 0].plot(t_y[: diag.size], diag, "b-", lw=1)
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
        return None
    return fig


def plot_simulated_data(
    c2_sim: np.ndarray,
    t: np.ndarray | None = None,
    phi_deg: float | None = None,
    contrast: float | None = None,
    offset: float | None = None,
    analysis_mode: AnalysisMode | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (8, 7),
    *,
    t2: np.ndarray | None = None,
    title: str | None = None,
) -> Figure | None:
    """Single-panel theoretical/fitted c2 heatmap with inline stats annotation.

    Used by the orchestrator to render fitted-only simulations (no comparison
    to experimental data). Annotates mean, range, and optional fit metadata
    (analysis_mode, contrast, offset).

    Parameters
    ----------
    c2_sim
        Theoretical or fitted c2 surface, shape ``(n_t1, n_t2)``.
    t
        Optional time axis (y / t₁).
    t2
        Optional x-axis (t₂). When supplied with ``t``, lets rectangular
        grids (n_t1 ≠ n_t2) render with the correct horizontal extent.
    phi_deg
        Optional phi angle for title.
    contrast, offset, analysis_mode
        Optional metadata annotations rendered in a corner box.
    title
        Optional base title override. Defaults to ``"Simulated C₂(t₁, t₂)"``.
        Pass e.g. ``"Experimental C₂(t₁, t₂)"`` when rendering real data.
        The ``φ=…`` suffix from ``phi_deg`` is appended regardless.
    save_path
        If provided, saved and closed; the function returns ``None``.
        Otherwise the live Figure is returned.
    figsize
        Matplotlib figsize in inches.

    Returns
    -------
    Figure or None
        The matplotlib Figure when ``save_path`` is ``None``; ``None`` when
        the figure was saved (and is therefore closed).
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Empty-input fallback — mirrors plot_nlsq_fit / plot_residual_map.
    if c2_sim.size == 0:
        fig.suptitle("No data available")
        if save_path is not None:
            _save_fig(fig, save_path)
            return None
        return fig

    n_t1, n_t2 = c2_sim.shape
    t1_vec = np.asarray(t) if t is not None else np.arange(n_t1, dtype=float)
    t2_vec = np.asarray(t2) if t2 is not None else (t1_vec if t is not None else np.arange(n_t2, dtype=float))
    # No transpose — rows=y=t1, cols=x=t2, consistent with plot_nlsq_fit and
    # plot_residual_map. The previous .T + swapped extent was inconsistent with
    # those functions on non-square grids (n_t1 ≠ n_t2).
    extent = (float(t2_vec[0]), float(t2_vec[-1]), float(t1_vec[0]), float(t1_vec[-1]))

    vmin, vmax = _resolve_color_limits(c2_sim, percentile_min=1.0, percentile_max=99.0)
    vmin = max(1.0, vmin)
    vmax = min(1.6, vmax) if vmax > 1.0 else vmax

    im = ax.imshow(
        c2_sim,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="jet",
        interpolation="bilinear",
        vmin=vmin,
        vmax=vmax,
    )
    base_title = title if title is not None else "Simulated C₂(t₁, t₂)"
    if phi_deg is not None:
        base_title = f"{base_title} at φ={phi_deg:.1f}°"
    ax.set_title(base_title, fontsize=13, fontweight="bold")
    ax.set_xlabel("t₂ (s)" if t is not None else "t₂ Index", fontsize=11)
    ax.set_ylabel("t₁ (s)" if t is not None else "t₁ Index", fontsize=11)
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
        return None
    return fig


_COMPRESSION_MAP = {
    "lzma": zipfile.ZIP_LZMA,
    "deflate": zipfile.ZIP_DEFLATED,
    "none": zipfile.ZIP_STORED,
}


def _write_npz_compressed(
    path: Path,
    arrays: Mapping[str, np.ndarray | np.floating | np.integer],
    *,
    compression: Literal["lzma", "deflate", "none"] = "lzma",
) -> None:
    """Write numerical arrays to .npz with configurable compression.

    Atomic rename: writes to a unique temp file via :func:`tempfile.mkstemp`
    in the same directory as ``path`` (so the rename is on the same
    filesystem), then renames over ``path``. The unique temp name lets
    concurrent calls targeting the same output coexist without clobbering
    each other's in-progress writes. Cleans up the temp file on any failure.

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
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temp file in the same directory — concurrent writers don't collide.
    fd, tmp_str = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_str)

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
                # Reject both plain ``object`` dtypes and structured dtypes that
                # contain object fields — ``np.lib.format.write_array`` will
                # otherwise fall back to a non-portable serializer, which the
                # ``allow_pickle=False`` reader path cannot load.
                if arr_np.dtype == object or getattr(arr_np.dtype, "hasobject", False):
                    raise TypeError(
                        f"array {name!r} has object dtype; NPZ requires numerical "
                        "arrays only (string metadata belongs in the JSON sidecar)"
                    )
                buf = io.BytesIO()
                np.lib.format.write_array(buf, arr_np)
                zf.writestr(f"{name}.npy", buf.getvalue())
        os.replace(tmp_path, path)
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
    analysis_mode: AnalysisMode,
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

    arrays: dict[str, np.ndarray | np.floating | np.integer] = {
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


def _plot_single_angle_datashader(args: tuple) -> Path:
    """Picklable worker: render one angle's 3-panel comparison via Datashader.

    Mirrors :func:`_render_one_angle_worker` but dispatches to the Datashader
    hybrid pipeline in :mod:`xpcsjax.viz.datashader_backend`. Used by the
    spawn-context Pool in :func:`_generate_plots_datashader` and reused
    inline for sequential fallback so the output is byte-identical
    regardless of which path produced it.
    """
    # Re-import inside the worker: spawn workers start cold and need a fresh
    # module import. The JAX env pin lives in xpcsjax/__init__.py and is
    # inherited from the parent's os.environ at spawn time.
    from xpcsjax.viz.datashader_backend import plot_c2_comparison_fast

    (
        phi_idx,
        c2_exp_i,
        c2_fit_i,
        residuals_i,
        t1,
        t2,
        phi_deg,
        output_dir,
        width,
        height,
        color_options,
    ) = args

    name_suffix = f"phi_{phi_idx:03d}_{phi_deg:.3f}deg"
    output_file = Path(output_dir) / f"c2_heatmaps_{name_suffix}.png"

    plot_c2_comparison_fast(
        np.asarray(c2_exp_i),
        np.asarray(c2_fit_i),
        np.asarray(residuals_i),
        np.asarray(t1),
        np.asarray(t2),
        output_file,
        phi_angle=phi_deg,
        width=width,
        height=height,
        **(color_options or {}),
    )
    return output_file


def _generate_plots_datashader(
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    c2_fitted: np.ndarray,
    residuals: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    output_dir: Path,
    *,
    parallel: bool = True,
    width: int = 1200,
    height: int = 1200,
    color_options: dict[str, Any] | None = None,
) -> None:
    """Render per-angle 3-panel comparisons via Datashader.

    Pool topology mirrors :mod:`homodyne.viz.nlsq_plots`: spawn-context
    ``multiprocessing.Pool`` initialised with :func:`_worker_init_cpu_only`,
    workers receive picklable per-angle tuples, on ``(OSError, RuntimeError,
    TimeoutError)`` the orchestrator catches and reruns the remaining work
    sequentially in the main process. This keeps the fast path under load
    spikes (Linux fork-bomb protection, transient HPC scheduler errors)
    without sacrificing the parallel speedup on the happy path.

    Angles whose ``c2_fitted`` is all-NaN are skipped (the per-angle compute
    failed upstream — no useful comparison to render).
    """
    n_phi = int(phi_angles.size)
    color_options = color_options or {}

    def _args_for(i: int) -> tuple:
        return (
            int(i),
            c2_exp[i],
            c2_fitted[i],
            residuals[i],
            t1,
            t2,
            float(phi_angles[i]),
            output_dir,
            width,
            height,
            color_options,
        )

    if parallel and n_phi > 1:
        try:
            ctx = multiprocessing.get_context("spawn")
            n_workers = min(multiprocessing.cpu_count(), n_phi)
            args_list = [
                _args_for(i) for i in range(n_phi) if not np.all(np.isnan(c2_fitted[i]))
            ]
            if not args_list:
                logger.warning(
                    "Datashader path: all angles have NaN c2_fitted; nothing to render"
                )
                return
            timeout_s = (60 * n_phi / max(n_workers, 1)) + 120
            with ctx.Pool(processes=n_workers, initializer=_worker_init_cpu_only) as pool:
                ar = pool.map_async(_plot_single_angle_datashader, args_list)
                ar.get(timeout=timeout_s)
            logger.info(
                "Datashader: rendered %d angles in parallel (%d workers)",
                len(args_list),
                n_workers,
            )
            return
        except Exception as e:
            logger.warning(
                "Parallel Datashader rendering failed (%s: %s); sequential fallback.",
                type(e).__name__,
                e,
            )
            logger.debug("Pool failure traceback:", exc_info=True)

    # Sequential path (use_datashader=True with parallel=False, n_phi==1,
    # or the parallel pool fell over).
    rendered = 0
    for i in range(n_phi):
        if np.all(np.isnan(c2_fitted[i])):
            continue
        _plot_single_angle_datashader(_args_for(i))
        rendered += 1
    logger.info("Datashader: rendered %d angles (sequential)", rendered)


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

    The JAX CPU pin lives in ``xpcsjax/__init__.py`` (env exported to spawn
    workers via the parent's ``os.environ``); per-worker re-assignment here
    would be too late because ``import jax.numpy`` at the top of this module
    has already run by the time the worker reaches this function.
    """
    from xpcsjax.viz.nlsq_plots import (
        plot_nlsq_fit,
        plot_residual_map,
        plot_simulated_data,
    )

    (
        phi_idx,
        c2_exp_i,
        c2_fit_i,
        t1,
        t2,
        phi_deg,
        plots,
        chi2_red,
        contrast,
        offset,
        analysis_mode,
        output_dir,
        sim_dir,
    ) = args

    # Filename includes the angle index so that .1f-equal angles
    # (e.g. 10.04° and 10.05°) don't collide under parallel rendering.
    name_suffix = f"phi_{phi_idx:03d}_{phi_deg:.3f}deg"

    if "comparison" in plots:
        plot_nlsq_fit(
            c2_exp_i,
            c2_fit_i,
            t=t1,
            t2=t2,
            phi_deg=phi_deg,
            reduced_chi_squared=chi2_red,
            save_path=Path(output_dir) / f"c2_heatmaps_{name_suffix}.png",
        )
    if "residuals" in plots:
        plot_residual_map(
            c2_exp_i,
            c2_fit_i,
            t=t1,
            t2=t2,
            phi_deg=phi_deg,
            save_path=Path(output_dir) / f"residuals_{name_suffix}.png",
        )
    if "simulated" in plots:
        plot_simulated_data(
            c2_fit_i,
            t=t1,
            t2=t2,
            phi_deg=phi_deg,
            contrast=contrast,
            offset=offset,
            analysis_mode=analysis_mode,
            save_path=Path(sim_dir) / f"simulated_c2_fitted_{name_suffix}.png",
        )


def generate_nlsq_plots(
    model: Any,
    result: Any,
    data: dict[str, Any],
    config: Any,
    output_dir: Path | str,
    *,
    use_datashader: bool = True,
    parallel: bool = True,
    plots: tuple[str, ...] = ("comparison", "residuals", "simulated"),
    compression: Literal["lzma", "deflate", "none"] = "lzma",
    datashader_width: int = 1200,
    datashader_height: int = 1200,
) -> None:
    """Generate NLSQ fit plots and serialize fitted artifacts.

    For each phi angle: recompute the fitted c2 surface via model dispatch,
    write PNG files for the selected plot families, then dump NPZ + JSON
    artifacts under ``output_dir/simulated_data/``.

    Parameters
    ----------
    model
        ``HomodyneModel`` or ``HeterodyneModel``. Heterodyne reads per-angle
        contrast/offset directly from ``result.parameters`` using the
        per-angle fit-time layout (Spec Amendment 3 resolution).
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
        If True (default) and the ``[viz-fast]`` extra is installed, render
        the 3-panel comparison plot via the Datashader hybrid pipeline (5-10x
        per-call speedup; in combination with ``parallel=True`` the cumulative
        speedup across many angles is ~50-200x). The matplotlib path is used
        as a transparent fallback when Datashader is missing. Mirrors
        homodyne's ``preview_mode`` semantics.
    parallel
        If True (default), render angles in a ``multiprocessing.Pool(spawn)``.
        The pool size is ``min(cpu_count(), n_phi)``. The matplotlib path
        honours this flag as well, but the speedup is much smaller because
        matplotlib's per-call cost is already low; the flag exists primarily
        to parallelize the Datashader path.
    plots
        Subset of ``{"comparison", "residuals", "simulated"}``. In Datashader
        mode only ``"comparison"`` is rendered via the fast path; residual
        diagnostics and simulated heatmaps need full matplotlib and are
        rendered via the matplotlib path **in addition** to the fast
        comparison when they appear in ``plots``.
    compression
        NPZ compression: ``"lzma"`` (default, best ratio), ``"deflate"``,
        or ``"none"``.
    datashader_width, datashader_height
        Per-panel rasterization resolution in pixels for the Datashader
        path. Default 1200×1200 (matches homodyne); reduce for faster
        rendering, increase for high-DPI publication output.

    Raises
    ------
    ValueError
        Unknown plot family, invalid compression, missing required data key,
        shape mismatch, or missing physics keys in config.
    TypeError
        Unsupported model type (not HomodyneModel, CombinedModel, or
        HeterodyneModel).
    """
    if not _is_supported_viz_model(model):
        raise TypeError(
            f"Unsupported model type: {type(model).__name__}. "
            f"Expected HomodyneModel, CombinedModel, or HeterodyneModel."
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
    analysis_mode = AnalysisMode.parse(
        str(config_dict.get("analysis_mode") or "laminar_flow")
    )

    # Output dirs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_dir = output_dir / "simulated_data"
    sim_dir.mkdir(parents=True, exist_ok=True)

    # Heterodyne: validate the per-angle scaling layout upfront. Without this,
    # an unsupported mode (fourier) would either silently produce all-NaN
    # artifacts (the per-angle compute loop catches Exception and leaves NaN)
    # or mis-infer n_phi from a residual that happens to be even. The
    # ``averaged`` and ``constant`` modes are now reconstructed by
    # ``_unpack_heterodyne_scaling`` (via ``per_angle_mode`` in diagnostics),
    # so only genuinely-unsupported layouts are rejected here.
    if _is_heterodyne_family(model):
        n_phi_expected = int(phi_angles.size)
        n_physical = len(model.parameter_names)
        n_total = int(np.asarray(result.parameters).size)
        individual_total = n_physical + 2 * n_phi_expected
        diagnostics = getattr(result, "nlsq_diagnostics", None) or {}
        per_angle_mode = diagnostics.get("per_angle_mode")
        supported_non_individual = per_angle_mode in ("averaged", "constant")
        if not supported_non_individual and n_total != individual_total:
            if n_total == n_physical:
                raise NotImplementedError(
                    f"Heterodyne 'constant' scaling mode is not yet supported by "
                    f"xpcsjax viz (got {n_physical} physical params with no "
                    f"per-angle scaling pairs and no 'constant' diagnostics). "
                    f"v0.1 supports per-angle 'individual', 'averaged', and "
                    f"'constant' modes. Use the upstream heterodyne package or "
                    f"wait for v0.2 for full mode parity."
                )
            raise NotImplementedError(
                f"Heterodyne result has {n_total} parameters but xpcsjax viz "
                f"expects {individual_total} (individual mode: {n_physical} "
                f"physics + 2*{n_phi_expected} per-angle scaling). Scaling mode "
                f"{per_angle_mode!r} (e.g. 'fourier') is not yet supported by "
                f"v0.1 viz."
            )

    # Per-model param unpacking (model-type dispatched inside helper).
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

    # Resolve backend choice. Datashader is the default fast path (matches
    # homodyne preview_mode). Missing-dep degrades silently to matplotlib so
    # callers without the [viz-fast] extra still get plots.
    use_ds = use_datashader and DATASHADER_AVAILABLE
    if use_datashader and not DATASHADER_AVAILABLE:
        logger.warning(
            "use_datashader=True but datashader is not installed. "
            "Install with: pip install 'xpcsjax[viz-fast]'. "
            "Falling back to matplotlib backend (publication quality)."
        )
    elif use_ds:
        logger.info("Using Datashader backend (fast preview rendering)")
    else:
        logger.info("Using matplotlib backend (publication quality)")

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

    # Residuals are needed by both backends and by the NPZ writer below.
    residuals = c2_exp - c2_fitted

    # Phase B: render PNGs. Backend dispatch follows the homodyne pattern —
    # Datashader handles only the 3-panel "comparison" (its strength is
    # large-array rasterization, not 4-panel diagnostics with histograms).
    # The matplotlib worker still handles "residuals" / "simulated" plot
    # families when those appear in ``plots``, regardless of backend choice.
    if use_ds and "comparison" in plots:
        _generate_plots_datashader(
            phi_angles=phi_angles,
            c2_exp=c2_exp,
            c2_fitted=c2_fitted,
            residuals=residuals,
            t1=t1,
            t2=t2,
            output_dir=output_dir,
            parallel=parallel,
            width=datashader_width,
            height=datashader_height,
        )
        # In Datashader mode the "comparison" plot family is satisfied by the
        # fast path. Drop it from the matplotlib plot set so we don't render
        # the 3-panel twice.
        mpl_plots: tuple[str, ...] = tuple(p for p in plots if p != "comparison")
    else:
        mpl_plots = plots

    if mpl_plots:
        # Matplotlib path renders whichever of {"comparison", "residuals",
        # "simulated"} the caller asked for AND that Datashader didn't
        # already cover. Reused for the no-Datashader fallback case.
        def _render_args_for_index(i: int) -> tuple:
            return (
                int(i),
                c2_exp[i],
                c2_fitted[i],
                t1,
                t2,
                float(phi_angles[i]),
                mpl_plots,
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
                    "Matplotlib: rendered %d angles in parallel (%d workers)",
                    len(args_list),
                    n_workers,
                )
            except Exception as e:
                logger.warning(
                    "Parallel rendering failed (%s: %s); sequential fallback.",
                    type(e).__name__,
                    e,
                )
                logger.debug("Pool failure traceback:", exc_info=True)
                for i in range(n_phi):
                    if np.all(np.isnan(c2_fitted[i])):
                        continue
                    _render_one_angle_worker(_render_args_for_index(i))
        else:
            for i in range(n_phi):
                if np.all(np.isnan(c2_fitted[i])):
                    continue
                _render_one_angle_worker(_render_args_for_index(i))

    # Slice uncertainties to match physical_params length.
    # Homodyne layout: [contrast, offset, physical...] -> skip first 2.
    # Heterodyne layout: [c_0..N-1, o_0..N-1, physical...] -> skip first 2*n_phi.
    n_phi_local = phi_angles.size
    if _is_heterodyne_family(model):
        skip = 2 * n_phi_local
    else:
        skip = 2
    phys_unc = np.asarray(result.uncertainties, dtype=float)[skip:]

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
        analysis_mode=analysis_mode,
        output_dir=sim_dir,
        compression=compression,
    )

    logger.info("NLSQ plot generation complete: %s", output_dir)
