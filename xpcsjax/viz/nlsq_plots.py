"""NLSQ fit visualization and artifact serialization.

This module is the matplotlib-based rendering layer for results produced by
:func:`xpcsjax.fit_nlsq`. It exposes four public entry points, all re-exported
from :mod:`xpcsjax.viz`:

- :func:`plot_nlsq_fit` — three-panel Experimental | Fitted | Residuals figure.
- :func:`plot_residual_map` — four-panel residual diagnostic figure.
- :func:`plot_simulated_data` — single-panel theoretical/fitted ``c2`` heatmap.
- :func:`generate_nlsq_plots` — orchestrator that recomputes per-angle fitted
  surfaces, writes PNGs, and serializes NPZ + JSON artifacts.

The three ``plot_*`` functions follow a common contract: they return the live
:class:`~matplotlib.figure.Figure` when ``save_path`` is ``None``, and return
``None`` (after saving and closing the figure) when ``save_path`` is provided.
:func:`generate_nlsq_plots` always returns ``None`` — it writes files instead.

Notes
-----
The module performs an optional-dependency probe for the Datashader fast-render
backend **before** its remaining imports; that ordering is intentional and is
exempted from import-sorting lint rules (``E402``/``I001``). Do not reorder.

See Also
--------
xpcsjax.viz.diagnostics : Diagonal-overlay statistics for fitted surfaces.
xpcsjax.fit_nlsq : Produces the :class:`OptimizationResult` these plots consume.
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
import matplotlib

# Pin the headless Agg backend BEFORE importing pyplot. This module renders
# PNGs to disk, including from multiprocessing "spawn" workers that cold-import
# it; without this pin a worker on a machine with DISPLAY could auto-select an
# interactive (Qt/Tk) backend and attempt GUI canvas creation off the main
# thread of a subprocess (a GUI-thread violation that can hang/crash).
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np

from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.io.json_utils import json_safe, json_serializer
from xpcsjax.utils.logging import get_logger
from xpcsjax.utils.path_validation import validate_plot_save_path

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
    # Percentile over finite values only — nanpercentile ignores NaN but lets a
    # single +/-inf skew the limits toward the extreme.
    finite = matrix[np.isfinite(matrix)]
    vmin = float(np.percentile(finite, percentile_min))
    vmax = float(np.percentile(finite, percentile_max))
    if not np.isfinite(vmin):
        vmin = 1.0
    if not np.isfinite(vmax):
        vmax = 1.5
    if vmin >= vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def _save_fig(fig: Figure, save_path: Path | str | None, dpi: int = 150) -> None:
    """Save figure to disk and close. No-op when ``save_path`` is None.

    The path is validated (traversal / null-byte / image-extension checks) via
    ``validate_plot_save_path`` before any directory is created or written, so a
    caller- or config-supplied path cannot escape to an arbitrary filesystem
    location. Creates parent directories as needed. The figure is closed even if
    ``savefig`` raises, so renderer/filesystem errors don't leak Figure handles.
    Logs the saved path (basename only) at INFO level.
    """
    if save_path is None:
        return
    # Validate, mkdir, and savefig are all inside the try so the figure is closed
    # even when validation rejects the path (no Figure-handle leak on any failure).
    # require_parent_exists=False because we create the parent ourselves below.
    try:
        p = validate_plot_save_path(save_path, require_parent_exists=False)
        if p is None:  # pragma: no cover - save_path is not None here
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    logger.info("Figure saved: %s", p.name)
    logger.debug("Figure full path: %s", p)


def _empty_data_fallback(fig: Figure, save_path: Path | str | None) -> Figure | None:
    """Render a "No data available" placeholder and apply the save/return contract.

    Shared by plot_nlsq_fit / plot_residual_map / plot_simulated_data's
    empty-input early return: set a suptitle, then either save-and-return-None
    (``save_path`` given) or return the live Figure (``save_path`` is None).
    """
    fig.suptitle("No data available")
    if save_path is not None:
        _save_fig(fig, save_path)
        return None
    return fig


def _resolve_extent(
    shape: tuple[int, int],
    t: np.ndarray | None,
    t2: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Derive ``(t1_vec, t2_vec, extent)`` for an imshow-transposed (n_t1, n_t2) surface.

    Shared by plot_nlsq_fit / plot_residual_map / plot_simulated_data: x = t1
    (horizontal), y = t2 (vertical), so rows -> t2 -> y and cols -> t1 -> x,
    with ``extent`` following the same (t1_min, t1_max, t2_min, t2_max) order.
    """
    n_t1, n_t2 = shape
    t1_vec = np.asarray(t) if t is not None else np.arange(n_t1, dtype=float)
    t2_vec = (
        np.asarray(t2)
        if t2 is not None
        else (t1_vec if t is not None else np.arange(n_t2, dtype=float))
    )
    extent = (float(t1_vec[0]), float(t1_vec[-1]), float(t2_vec[0]), float(t2_vec[-1]))
    return t1_vec, t2_vec, extent


def _is_homodyne_family(model: Any) -> bool:
    """Return ``True`` for models using the homodyne result layout.

    The homodyne layout is ``[contrast, offset, *physical]`` with scalar
    per-angle scaling. Two concrete types qualify: :class:`HomodyneModel` (the stateful viz
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
    """Return ``True`` for :class:`HeterodyneModel` (per-angle ``[c.., o.., *physical]``)."""
    from xpcsjax.core.heterodyne_model import HeterodyneModel

    return isinstance(model, HeterodyneModel)


def _is_supported_viz_model(model: Any) -> bool:
    """Return ``True`` for any model type the viz layer knows how to plot."""
    return _is_homodyne_family(model) or _is_heterodyne_family(model)


def _homodyne_scaling_arrays(
    model: Any, result: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return per-angle scaling and physical params for a homodyne-family result.

    The four returned arrays are ``(contrasts, offsets, physical_params,
    names)``. The homodyne NLSQ fit uses per-angle scaling by default, so
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


def _heterodyne_physics_is_head(diagnostics: dict[str, Any]) -> bool:
    """Whether the heterodyne physics block is the HEAD of the param/uncertainty vector.

    Single source of truth for the layout decision so the physical *params* slice
    (:func:`_unpack_result_params` / :func:`_unpack_heterodyne_scaling`) and the
    physical *uncertainty* slice (:func:`generate_nlsq_plots`) can never diverge:

    - ``averaged`` — physics-first iff ``scaling_first`` is ``False``. As of the
      tied-parameters PR, ``_fit_joint_averaged_multi_phi`` always permutes its
      result to canonical scaling-first and sets ``scaling_first=True`` before
      returning, so this branch is not currently produced by any in-tree
      caller; it stays honoured defensively for a marker-provided ``False``.
      Scaling-first (physics trailing) otherwise, including the marker-less
      default.
    - ``constant`` — physics-only vector; head and tail coincide (report head).
    - ``individual`` / marker-less — canonical scaling-first → physics is the TAIL.
    """
    mode = diagnostics.get("per_angle_mode")
    if mode == "averaged":
        return not bool(diagnostics.get("scaling_first", True))
    if mode == "constant":
        return True
    return False


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
        contrasts, offsets, physical_params, names = _homodyne_scaling_arrays(model, result)
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
        # Averaged mode: canonical scaling-first layout [contrast, offset, physics...].
        # Scalar summary is the single fitted pair (physics is the trailing 14-vector).
        if mode == "averaged":
            # Two producers emit averaged with OPPOSITE orderings (audit #1):
            # the engine route is SCALING-FIRST ([c, o | physics]) and the legacy
            # _fit_joint_averaged_multi_phi is PHYSICS-FIRST ([physics | c, o]).
            # Honour the explicit scaling_first marker; default True (canonical
            # scaling-first) for marker-less results. Scaling scalars come from
            # diagnostics regardless of layout; only the physics slice and the
            # scalar fallback depend on the layout.
            if _heterodyne_physics_is_head(diagnostics):
                physical_params = params[:n_physical].copy()
                fallback_c, fallback_o = params[-2], params[-1]
            else:
                physical_params = params[-n_physical:].copy()
                fallback_c, fallback_o = params[0], params[1]
            contrast_scalar = float(diagnostics.get("averaged_contrast", fallback_c))
            offset_scalar = float(diagnostics.get("averaged_offset", fallback_o))
            return contrast_scalar, offset_scalar, physical_params, physical_names
        # Constant mode: [physics...] only (physics-only vector); scaling frozen in
        # diagnostics — params[:n_physical] is the entire vector.
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
        # Canonical scaling-first layout: [c_0..N-1, o_0..N-1, physical_0..n_physical-1]
        # — the scaling HEAD precedes the physics TAIL (Tasks 3-12 of per-angle-mode
        # unification). Require 2*n_phi + n_physical params; the residual
        # (n_total - n_physical) must be even and non-negative. We tolerate the
        # no-scaling case (n_total == n_physical) by returning zero scalars so the
        # downstream simulated-data annotation panel still renders; the real
        # heterodyne per-angle evaluation path uses ``_unpack_heterodyne_scaling``
        # which fails loudly for that case.
        residual = n_total - n_physical
        if residual < 0 or residual % 2 != 0:
            raise ValueError(
                f"HeterodyneModel expects 2*n_phi + {n_physical} params "
                f"(per-angle layout); got {n_total}. The residual "
                f"{residual} is not divisible by 2."
            )
        n_phi = residual // 2
        physical_params = params[-n_physical:].copy()
        contrasts = params[:n_phi]
        offsets = params[n_phi : 2 * n_phi]
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
      contrast/offset fitted independently). Canonical scaling-first: the
      scaling HEAD precedes the physics TAIL.
    - ``averaged`` — ``[contrast, offset, physical...]``; the single fitted
      (contrast, offset) pair is replicated across all angles.
      ``_fit_joint_averaged_multi_phi`` always permutes its result to this
      canonical scaling-first layout and sets ``scaling_first=True`` before
      returning (as of the tied-parameters PR) — see
      :func:`_heterodyne_physics_is_head`, the single source of truth for
      which end of the vector the physics block occupies, for the
      (currently unused in-tree) physics-first fallback it still honours.
    - ``constant`` — ``[physical...]``; per-angle scaling was frozen pre-fit
      and is read from the ``contrast_per_angle_fixed`` /
      ``offset_per_angle_fixed`` diagnostics.

    When ``nlsq_diagnostics`` is absent (e.g. a synthetic result), the layout
    is inferred from the parameter count and treated as ``individual``. The
    caller must supply ``n_phi_expected`` (from ``data["phi_angles_list"]``) so
    the individual layout can be disambiguated from any unrecognised layout
    whose extra slots would otherwise be misread as ``2*n_phi``.

    Returns
    -------
    (contrasts, offsets, physical_params, n_phi)
        contrasts, offsets: shape (n_phi_expected,)
        physical_params:    shape (n_physical=14,)
        n_phi:              equals ``n_phi_expected``

    Raises
    ------
    NotImplementedError
        An unrecognised per-angle scaling mode, or a diagnostics-less result
        whose parameter count matches no recognised layout. Those remain out of
        scope for viz.
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

    # Averaged mode: canonical scaling-first layout is ``[contrast, offset,
    # physics...]`` — one fitted (contrast, offset) pair shared across all
    # angles. Prefer the fitted scalars stored in diagnostics; fall back to
    # the leading two scaling-HEAD slots (params[0] / params[1] — the physics
    # tail occupies params[-n_physical:]). Replicate across n_phi so the
    # per-angle evaluation path stays uniform with individual mode.
    if mode == "averaged":
        # Marker-aware layout (audit #1): every current producer (engine
        # route and _fit_joint_averaged_multi_phi alike) emits SCALING-FIRST
        # and sets scaling_first=True. Default True (canonical scaling-first)
        # when the marker is absent; the physics-first branch below stays
        # honoured defensively should a future/marker-provided producer emit
        # scaling_first=False.
        if _heterodyne_physics_is_head(diagnostics):
            physical_params = params[:n_physical].copy()
            fallback_c, fallback_o = params[-2], params[-1]
        else:
            physical_params = params[-n_physical:].copy()
            fallback_c, fallback_o = params[0], params[1]
        contrast = float(diagnostics.get("averaged_contrast", fallback_c))
        offset = float(diagnostics.get("averaged_offset", fallback_o))
        contrasts = np.full(n_phi_expected, contrast, dtype=float)
        offsets = np.full(n_phi_expected, offset, dtype=float)
        return contrasts, offsets, physical_params, n_phi_expected

    # Constant mode: layout is ``[physics...]`` only (physics-only vector, no
    # scaling head in result.parameters); per-angle scaling was frozen pre-fit
    # and is carried in diagnostics.
    if mode == "constant":
        physical_params = params[:n_physical].copy()
        contrasts = np.asarray(diagnostics["contrast_per_angle_fixed"], dtype=float).ravel()
        offsets = np.asarray(diagnostics["offset_per_angle_fixed"], dtype=float).ravel()
        if contrasts.size != n_phi_expected or offsets.size != n_phi_expected:
            raise ValueError(
                f"Constant-mode per-angle scaling has {contrasts.size} contrasts / "
                f"{offsets.size} offsets but {n_phi_expected} angles were requested."
            )
        return contrasts, offsets, physical_params, n_phi_expected

    # Individual mode (or a diagnostics-less result, e.g. a synthetic
    # OptimizationResult): canonical scaling-first layout is
    # ``[c_0..N-1, o_0..N-1, physics...]`` — the scaling HEAD precedes the
    # physics TAIL (Tasks 3-12 of per-angle-mode unification).
    individual_total = n_physical + 2 * n_phi_expected
    if n_total != individual_total:
        if n_total == n_physical:
            raise NotImplementedError(
                f"Heterodyne 'constant' scaling mode is not yet supported by xpcsjax "
                f"viz. Only per-angle 'individual' mode is supported — got "
                f"{n_physical} physical params with no per-angle (contrast, offset) "
                f"pairs in result.parameters. Use the upstream heterodyne package or "
                f"wait for full mode parity."
            )
        raise NotImplementedError(
            f"Heterodyne result has {n_total} parameters but xpcsjax viz expects "
            f"{individual_total} (individual mode: {n_physical} physics + "
            f"2*{n_phi_expected} per-angle scaling). Scaling mode "
            f"{mode!r} is not yet supported by viz."
        )
    physical_params = params[-n_physical:].copy()
    contrasts = params[:n_phi_expected].copy()
    offsets = params[n_phi_expected : 2 * n_phi_expected].copy()
    return contrasts, offsets, physical_params, n_phi_expected


def _resolve_phi_index(data: dict[str, Any], phi_deg: float, phi_index: int | None) -> int:
    """Resolve the per-angle index for ``phi_deg``.

    Prefer the caller's loop index (``phi_index``) when provided — it is
    authoritative and unambiguous even when ``phi_angles_list`` contains two
    entries with the same nominal value. Fall back to a value-based
    ``np.isclose`` lookup only when the index is not known (e.g. an external
    caller that has just a phi value).
    """
    if phi_index is not None:
        return int(phi_index)
    phi_array = np.asarray(data["phi_angles_list"], dtype=float)
    matches = np.where(np.isclose(phi_array, phi_deg, atol=1e-6))[0]
    if matches.size == 0:
        raise ValueError(
            f"phi_deg={phi_deg!r} not found in data['phi_angles_list'] "
            f"(values: {phi_array.tolist()})"
        )
    return int(matches[0])


def _evaluate_c2_per_angle(
    model: Any,
    result: Any,
    data: dict[str, Any],
    config: dict[str, Any],
    phi_deg: float,
    phi_index: int | None = None,
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
            i = _resolve_phi_index(data, phi_deg, phi_index)
            contrast = float(contrasts[i])
            offset = float(offsets[i])
        # HomodyneModel (the stateful wrapper) carries pre-computed grids /
        # physics-factors and exposes a single-angle helper. The bare
        # CombinedModel that ``make_model`` returns for static_*/laminar_flow
        # does not, so drive its ``compute_g2`` with q/L/dt from the config and
        # the data's time grids (``compute_g2`` applies ``offset + contrast*g1**2``
        # internally — mirrors plot_families.simulated._evaluate_model_c2). Capability
        # dispatch rather than isinstance-per-type keeps any future
        # homodyne-family model working as long as it exposes one of these APIs.
        if hasattr(model, "compute_c2_single_angle"):
            c2 = model.compute_c2_single_angle(physical_params, phi_deg, contrast, offset)
            return np.asarray(c2)
        ap = config.get("analyzer_parameters") or {}
        q_raw = (ap.get("scattering") or {}).get("wavevector_q")
        if q_raw is None:
            raise ValueError("Missing analyzer_parameters.scattering.wavevector_q")
        L_raw = (ap.get("geometry") or {}).get("stator_rotor_gap")
        if L_raw is None:
            raise ValueError("Missing analyzer_parameters.geometry.stator_rotor_gap")
        # Explicit None-check so dt=0 is not treated as falsy.
        dt_raw = ap.get("dt")
        if dt_raw is None:
            dt_raw = (ap.get("temporal") or {}).get("dt")
        if dt_raw is None:
            raise ValueError("Missing analyzer_parameters: 'dt' or 'temporal.dt' is required")
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
        i = _resolve_phi_index(data, phi_deg, phi_index)
        n_phi_expected = int(phi_array.size)
        contrasts, offsets, physical_params, _ = _unpack_heterodyne_scaling(
            model, result, n_phi_expected=n_phi_expected
        )

        ap = config.get("analyzer_parameters") or {}
        q_raw = (ap.get("scattering") or {}).get("wavevector_q")
        if q_raw is None:
            raise ValueError("Missing analyzer_parameters.scattering.wavevector_q")
        q = float(q_raw)
        L_raw = (ap.get("geometry") or {}).get("stator_rotor_gap")
        if L_raw is None:
            raise ValueError("Missing analyzer_parameters.geometry.stator_rotor_gap")
        L = float(L_raw)
        # Use explicit None-check so dt=0 is not treated as falsy.
        dt_raw = ap.get("dt")
        if dt_raw is None:
            dt_raw = (ap.get("temporal") or {}).get("dt")
        if dt_raw is None:
            raise ValueError("Missing analyzer_parameters: 'dt' or 'temporal.dt' is required")
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
        Optional time axis (seconds) — used as the x-axis (t₁). If ``t2``
        is also ``None``, the same vector is used for both axes (square
        assumption). If ``None``, uses index axes.
    t2
        Optional y-axis (t₂). When supplied with ``t``, lets rectangular
        grids (n_t1 ≠ n_t2) render with the correct vertical extent.
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
        the figure was saved (and is therefore closed). When either input is
        empty, a single-panel "No data available" figure is returned (or saved)
        instead of the three-panel layout.

    See Also
    --------
    plot_residual_map : Four-panel residual diagnostic for the same surfaces.
    plot_simulated_data : Single-panel heatmap for a fitted-only surface.
    generate_nlsq_plots : Orchestrator that calls this per angle.

    Examples
    --------
    >>> import numpy as np
    >>> from xpcsjax.viz import plot_nlsq_fit
    >>> c2_exp = np.full((64, 64), 1.2)
    >>> c2_fit = c2_exp + np.random.default_rng(0).normal(0, 1e-3, c2_exp.shape)
    >>> fig = plot_nlsq_fit(c2_exp, c2_fit, reduced_chi_squared=1.05)
    >>> fig is not None  # live Figure returned when save_path is None
    True
    >>> plot_nlsq_fit(c2_exp, c2_fit, save_path="fit.png")  # saved, returns None
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    try:

        if c2_exp.size == 0 or c2_fit.size == 0:
            return _empty_data_fallback(fig, save_path)

        # x = t₁ (horizontal), y = t₂ (vertical): the (n_t1, n_t2) surfaces are
        # transposed at imshow so rows→t₂→y and cols→t₁→x, with extent following.
        _, _, extent = _resolve_extent(c2_exp.shape, t, t2)

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
            c2_exp.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="jet",
            vmin=vmin_shared,
            vmax=vmax_shared,
        )
        axes[0].set_box_aspect(1)
        axes[0].set_title(f"Experimental Data{phi_str}")
        axes[0].set_xlabel("t₁")
        axes[0].set_ylabel("t₂")
        plt.colorbar(im0, ax=axes[0], label="c₂")

        im1 = axes[1].imshow(
            c2_fit.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="jet",
            vmin=vmin_shared,
            vmax=vmax_shared,
        )
        axes[1].set_box_aspect(1)
        axes[1].set_title(f"Fitted Model{phi_str}")
        axes[1].set_xlabel("t₁")
        axes[1].set_ylabel("t₂")
        plt.colorbar(im1, ax=axes[1], label="c₂")

        residual = c2_exp - c2_fit
        finite_r = residual[np.isfinite(residual)]
        vmax_r = float(np.nanpercentile(np.abs(finite_r), 99)) if finite_r.size > 0 else 1.0
        if vmax_r == 0.0 or not np.isfinite(vmax_r):
            vmax_r = 1.0
        im2 = axes[2].imshow(
            residual.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax_r,
            vmax=vmax_r,
        )
        axes[2].set_box_aspect(1)
        axes[2].set_title(f"Residuals{phi_str}")
        axes[2].set_xlabel("t₁")
        axes[2].set_ylabel("t₂")
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
    except Exception:
        plt.close(fig)
        raise


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
        Optional time axis (x / t₁). Falls back to index axis when None.
    t2
        Optional y-axis (t₂). When supplied with ``t``, lets rectangular
        grids (n_t1 ≠ n_t2) render with the correct vertical extent.
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
        the figure was saved (and is therefore closed). When either input is
        empty, a "No data available" figure is returned (or saved) instead.

    See Also
    --------
    plot_nlsq_fit : Three-panel Experimental | Fitted | Residuals comparison.
    compute_diagonal_overlay_stats : Numerical t₁=t₂ diagonal statistics.
    generate_nlsq_plots : Orchestrator that calls this per angle.

    Examples
    --------
    >>> import numpy as np
    >>> from xpcsjax.viz import plot_residual_map
    >>> c2_exp = np.full((64, 64), 1.2)
    >>> c2_fit = c2_exp + np.random.default_rng(0).normal(0, 1e-3, c2_exp.shape)
    >>> fig = plot_residual_map(c2_exp, c2_fit, phi_deg=45.0)
    >>> fig is not None
    True
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    try:

        if c2_exp.size == 0 or c2_fit.size == 0:
            return _empty_data_fallback(fig, save_path)

        residuals = c2_exp - c2_fit
        # x = t₁ (horizontal), y = t₂ (vertical): transpose the (n_t1, n_t2) map.
        t1_vec, _, extent = _resolve_extent(residuals.shape, t, t2)

        # [0,0] Residual Map
        finite_r = residuals[np.isfinite(residuals)]
        vmax = float(np.nanpercentile(np.abs(finite_r), 99)) if finite_r.size > 0 else 1.0
        if vmax == 0.0 or not np.isfinite(vmax):
            vmax = 1.0
        im = axes[0, 0].imshow(
            residuals.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        axes[0, 0].set_box_aspect(1)
        axes[0, 0].set_title("Residual Map")
        axes[0, 0].set_xlabel("t₁")
        axes[0, 0].set_ylabel("t₂")
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
        axes[0, 1].set_box_aspect(1)
        axes[0, 1].set_xlabel("Residual Value")
        axes[0, 1].set_ylabel("Density")
        axes[0, 1].set_title("Residual Distribution")
        # Stats over the already-computed finite residuals — nanmean/nanstd ignore
        # NaN but not inf, and flat_finite has both excluded.
        mu = float(np.mean(flat_finite)) if flat_finite.size > 0 else 0.0
        sigma = float(np.std(flat_finite)) if flat_finite.size > 0 else 0.0
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

        # [1,0] Diagonal residuals — length is min(n_t1, n_t2); plot against t1 truncated.
        diag = np.diag(residuals)
        axes[1, 0].plot(t1_vec[: diag.size], diag, "b-", lw=1)
        axes[1, 0].axhline(0, color="k", linestyle="--", alpha=0.5)
        axes[1, 0].set_box_aspect(1)
        axes[1, 0].set_xlabel("Time")
        axes[1, 0].set_ylabel("Residual")
        axes[1, 0].set_title("Diagonal Residuals")

        # [1,1] Residuals vs Fitted. A single angle can carry 10^7-10^8 points
        # (datashader_backend.py's module docstring); an unstrided PathCollection
        # over that many points allocates ~GB-scale offsets/color arrays per
        # worker before rendering even starts. Subsample with a fixed seed
        # (reproducibility) so the scatter stays representative, not the full set.
        fitted_flat = c2_fit.ravel()
        residuals_flat = residuals.ravel()
        _MAX_RESIDUAL_SCATTER_POINTS = 50_000
        if fitted_flat.size > _MAX_RESIDUAL_SCATTER_POINTS:
            rng = np.random.default_rng(42)
            idx = rng.choice(fitted_flat.size, size=_MAX_RESIDUAL_SCATTER_POINTS, replace=False)
            fitted_flat = fitted_flat[idx]
            residuals_flat = residuals_flat[idx]
        axes[1, 1].scatter(fitted_flat, residuals_flat, alpha=0.1, s=1)
        axes[1, 1].axhline(0, color="r", linestyle="--")
        axes[1, 1].set_box_aspect(1)
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
    except Exception:
        plt.close(fig)
        raise


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
        Optional time axis (x / t₁).
    t2
        Optional y-axis (t₂). When supplied with ``t``, lets rectangular
        grids (n_t1 ≠ n_t2) render with the correct vertical extent.
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
        the figure was saved (and is therefore closed). When ``c2_sim`` is
        empty, a "No data available" figure is returned (or saved) instead.

    See Also
    --------
    plot_nlsq_fit : Compare a fitted surface against experimental data.
    generate_nlsq_plots : Orchestrator that calls this per angle.

    Examples
    --------
    >>> import numpy as np
    >>> from xpcsjax.viz import plot_simulated_data
    >>> c2_sim = np.full((64, 64), 1.25)
    >>> fig = plot_simulated_data(
    ...     c2_sim, phi_deg=0.0, contrast=0.25, offset=1.0,
    ...     title="Experimental C₂(t₁, t₂)",
    ... )
    >>> fig is not None
    True
    """
    fig, ax = plt.subplots(figsize=figsize)
    try:

        # Empty-input fallback — mirrors plot_nlsq_fit / plot_residual_map.
        if c2_sim.size == 0:
            return _empty_data_fallback(fig, save_path)

        # x = t₁ (horizontal), y = t₂ (vertical): transpose the (n_t1, n_t2) surface
        # so rows→t₂→y and cols→t₁→x, consistent with plot_nlsq_fit and
        # plot_residual_map (which also transpose + use a (t₁, t₂) extent).
        _, _, extent = _resolve_extent(c2_sim.shape, t, t2)

        vmin, vmax = _resolve_color_limits(c2_sim, percentile_min=1.0, percentile_max=99.0)
        vmin = max(1.0, vmin)
        vmax = min(1.6, vmax) if vmax > 1.0 else vmax
        if vmin >= vmax:
            # Degenerate sub-1.0 surface: avoid passing vmin > vmax (inverted colormap)
            vmax = vmin + 0.5

        im = ax.imshow(
            c2_sim.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="jet",
            interpolation="bilinear",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_box_aspect(1)
        base_title = title if title is not None else "Simulated C₂(t₁, t₂)"
        if phi_deg is not None:
            base_title = f"{base_title} at φ={phi_deg:.1f}°"
        ax.set_title(base_title, fontsize=13, fontweight="bold")
        ax.set_xlabel("t₁ (s)" if t is not None else "t₁ Index", fontsize=11)
        ax.set_ylabel("t₂ (s)" if t is not None else "t₂ Index", fontsize=11)
        cbar = plt.colorbar(im, ax=ax, label="C₂", shrink=0.9)
        cbar.ax.tick_params(labelsize=9)

        finite = c2_sim[np.isfinite(c2_sim)]
        if finite.size > 0:
            # Use the finite subset, not raw c2_sim: nanmean/nanmin/nanmax only
            # skip NaN, not inf, so a single inf would poison the annotation.
            mean_v = float(np.mean(finite))
            min_v = float(np.min(finite))
            max_v = float(np.max(finite))
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
    except Exception:
        plt.close(fig)
        raise


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

    # Unique temp file in the same directory (mirrors _write_npz_compressed
    # above) — concurrent writers targeting the same output_dir don't collide.
    fd, tmp_str = tempfile.mkstemp(
        prefix=json_path.name + ".", suffix=".tmp", dir=str(json_path.parent)
    )
    os.close(fd)
    tmp_json = Path(tmp_str)
    try:
        with open(tmp_json, "w", encoding="utf-8") as f:
            # json_safe() converts NaN/Infinity to RFC-8259-legal null/strings
            # so non-converged or all-NaN fits still yield strictly parseable JSON.
            json.dump(json_safe(meta), f, indent=2, default=json_serializer)
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
            args_list = [_args_for(i) for i in range(n_phi) if not np.all(np.isnan(c2_fitted[i]))]
            if not args_list:
                logger.warning("Datashader path: all angles have NaN c2_fitted; nothing to render")
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
        except (OSError, RuntimeError, multiprocessing.TimeoutError) as e:
            # NOTE: this module only ``import multiprocessing`` (no
            # ``from multiprocessing import TimeoutError``), so the bare
            # ``TimeoutError`` this docstring names is
            # ``multiprocessing.TimeoutError`` (what ``ar.get(timeout=...)``
            # actually raises) -- NOT the unrelated builtin ``TimeoutError``.
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
    """Pool worker initializer — pin JAX to CPU + lazy allocator + headless mpl."""
    import os

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    # Belt-and-suspenders headless pin for any matplotlib import path in the
    # worker (the module-top matplotlib.use("Agg") already covers nlsq_plots).
    os.environ["MPLBACKEND"] = "Agg"


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
        :class:`~xpcsjax.HomodyneModel`, the bare ``CombinedModel`` returned by
        :func:`xpcsjax.core.models.make_model` for the homodyne modes, or
        :class:`~xpcsjax.HeterodyneModel`. The two families use different
        ``result.parameters`` layouts, and heterodyne's varies by per-angle
        mode. The per-model unpacking is dispatched internally, so callers do
        not need to reconcile the layouts; code that must slice
        ``result.parameters`` directly should consult
        :func:`_heterodyne_physics_is_head` rather than assume an ordering.
    result
        :class:`~xpcsjax.OptimizationResult` from :func:`xpcsjax.fit_nlsq`.
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
        If True (default), render the 3-panel comparison plot via the
        Datashader hybrid pipeline (5-10x per-call speedup; in combination
        with ``parallel=True`` the cumulative speedup across many angles is
        ~50-200x). Datashader is a core dependency, always installed; the
        matplotlib path is used as a transparent fallback only in a
        broken/incomplete environment where it is missing. Mirrors
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

    Returns
    -------
    None
        This function writes PNG, NPZ, and JSON files under ``output_dir`` and
        ``output_dir/simulated_data/``; it does not return the figures. Use the
        ``plot_*`` functions directly if you need live Figure objects.

    Raises
    ------
    ValueError
        Unknown plot family, invalid compression, missing required data key,
        shape mismatch, or missing physics keys in config.
    TypeError
        Unsupported model type (not HomodyneModel, CombinedModel, or
        HeterodyneModel).
    NotImplementedError
        Heterodyne result whose per-angle scaling layout is unsupported by
        viz (e.g. a ``constant`` layout without the matching ``per_angle_mode``
        diagnostics). Per-angle ``individual``, ``averaged``, and
        reconstructable ``constant`` modes are supported.

    See Also
    --------
    plot_nlsq_fit : Per-angle comparison figure rendered by this orchestrator.
    plot_residual_map : Per-angle residual diagnostic figure.
    plot_simulated_data : Per-angle fitted-surface heatmap.
    xpcsjax.fit_nlsq : Produces the ``result`` consumed here.

    Examples
    --------
    >>> from xpcsjax import fit_nlsq, ConfigManager  # doctest: +SKIP
    >>> from xpcsjax.viz import generate_nlsq_plots  # doctest: +SKIP
    >>> result = fit_nlsq(...)  # doctest: +SKIP
    >>> generate_nlsq_plots(  # doctest: +SKIP
    ...     model, result, data, config, "outputs/run01",
    ...     plots=("comparison", "residuals"),
    ... )
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

    ap = config_dict.get("analyzer_parameters") or {}
    q = (ap.get("scattering") or {}).get("wavevector_q")
    L = (ap.get("geometry") or {}).get("stator_rotor_gap")
    dt = ap.get("dt")
    if dt is None:
        # Fall back to temporal.dt — homodyne configs nest it there.
        dt = (ap.get("temporal") or {}).get("dt")
    if q is None or L is None or dt is None:
        raise ValueError(
            "config.analyzer_parameters must contain scattering.wavevector_q, "
            "geometry.stator_rotor_gap, and dt (or temporal.dt)"
        )
    analysis_mode = AnalysisMode.parse(str(config_dict.get("analysis_mode") or "laminar_flow"))

    # Output dirs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_dir = output_dir / "simulated_data"
    sim_dir.mkdir(parents=True, exist_ok=True)

    # Heterodyne: validate the per-angle scaling layout upfront. Without this,
    # an unsupported mode would either silently produce all-NaN
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
        if supported_non_individual:
            # ``model`` and ``result`` are independently-suppliable args (e.g. a
            # cached result paired with a freshly-constructed model), so a size
            # floor is still needed here: without it a too-short params vector
            # silently truncates physical_params and feeds JAX's clip-not-raise
            # out-of-bounds indexing, producing a wrong-but-finite fitted surface.
            n_physical_floor = n_physical + 2 if per_angle_mode == "averaged" else n_physical
            if n_total < n_physical_floor:
                raise NotImplementedError(
                    f"Heterodyne result has {n_total} parameters but "
                    f"{per_angle_mode!r} scaling mode needs at least "
                    f"{n_physical_floor} ({n_physical} physics"
                    f"{' + 2 averaged scaling' if per_angle_mode == 'averaged' else ''})."
                    " The (model, result) pair looks mismatched."
                )
        if not supported_non_individual and n_total != individual_total:
            if n_total == n_physical:
                raise NotImplementedError(
                    f"Heterodyne 'constant' scaling mode is not yet supported by "
                    f"xpcsjax viz (got {n_physical} physical params with no "
                    f"per-angle scaling pairs and no 'constant' diagnostics). "
                    f"Per-angle 'individual', 'averaged', and "
                    f"'constant' modes are supported. Use the upstream heterodyne package or "
                    f"wait for full mode parity."
                )
            raise NotImplementedError(
                f"Heterodyne result has {n_total} parameters but xpcsjax viz "
                f"expects {individual_total} (individual mode: {n_physical} "
                f"physics + 2*{n_phi_expected} per-angle scaling). Scaling mode "
                f"{per_angle_mode!r} is not yet supported by viz."
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
    # homodyne preview_mode) and a CORE dependency (not an optional extra);
    # missing-dep degrades silently to matplotlib so callers in a
    # broken/incomplete environment still get plots.
    use_ds = use_datashader and DATASHADER_AVAILABLE
    if use_datashader and not DATASHADER_AVAILABLE:
        logger.warning(
            "use_datashader=True but datashader is not installed (it is a "
            "core xpcsjax dependency -- this indicates a broken/incomplete "
            "install). Reinstall with: pip install xpcsjax. "
            "Falling back to matplotlib backend (publication quality)."
        )
    elif use_ds:
        logger.info("Using Datashader backend (fast preview rendering)")
    else:
        logger.info("Using matplotlib backend (publication quality)")

    # Phase A: compute fitted surfaces in main process (models may not be picklable)
    chi2_red = float(result.reduced_chi_squared)
    failed_angle_indices: list[int] = []
    for i, phi_deg in enumerate(phi_angles):
        phi_deg_f = float(phi_deg)
        try:
            c2_fitted[i] = _evaluate_c2_per_angle(
                model, result, data, config_dict, phi_deg_f, phi_index=i
            )
        except Exception:
            logger.exception(
                "Angle %d (phi=%.1f) compute failed; leaving NaN in c2_fitted",
                i,
                phi_deg_f,
            )
            failed_angle_indices.append(i)

    if np.all(np.isnan(c2_fitted)):
        raise RuntimeError(
            f"Model evaluation failed for all {n_phi} angle(s); see prior "
            "ERROR log entries for per-angle tracebacks. No artifacts written."
        )

    # Count only angles whose evaluation actually raised, not any angle that
    # merely contains a NaN value -- a successful evaluation can legitimately
    # produce some NaN entries (e.g. degenerate physics at extreme parameter
    # values) without having failed.
    n_failed_angles = len(failed_angle_indices)
    if 0 < n_failed_angles < n_phi:
        logger.warning(
            "%d of %d angle(s) failed to evaluate; artifacts contain NaN "
            "gaps for those angles (see prior ERROR log entries above).",
            n_failed_angles,
            n_phi,
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
            except (OSError, RuntimeError, multiprocessing.TimeoutError) as e:
                # See _generate_plots_datashader's mirror block: the
                # ``multiprocessing.TimeoutError`` ``ar.get(timeout=...)``
                # raises is a distinct class from the unrelated builtin
                # ``TimeoutError`` -- this module never shadows the name.
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

    # Slice uncertainties to match physical_params — using the SAME layout
    # decision as _unpack_result_params so values and uncertainties can never
    # diverge (the 2026-06-21 twin-path bug). Homodyne is always scaling-first
    # ([c_0..N-1, o_0..N-1, physical...]) so its physics uncertainties trail.
    # Heterodyne honours _heterodyne_physics_is_head: physics-first averaged
    # (scaling_first=False) and constant put physics at the HEAD; scaling-first
    # averaged / individual put it at the TAIL.
    all_unc = np.asarray(result.uncertainties, dtype=float)
    if _is_heterodyne_family(model):
        n_physical_het = len(model.parameter_names)
        diagnostics = getattr(result, "nlsq_diagnostics", None) or {}
        if _heterodyne_physics_is_head(diagnostics):
            phys_unc = all_unc[:n_physical_het]
        else:
            phys_unc = all_unc[-n_physical_het:]
    else:
        # Use the already-extracted physical-parameter count (HomodyneModel has
        # no ``parameter_names`` attribute; physical_params is resolved robustly
        # by _unpack_result_params for both model families).
        n_physical = len(physical_params)
        phys_unc = all_unc[-n_physical:]

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
