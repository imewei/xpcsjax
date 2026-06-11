"""Plot generation dispatch for the xpcsjax CLI.

NLSQ-only by design — xpcsjax does not ship Bayesian/CMC sampling. This module
fans the parsed CLI args out to the relevant ``xpcsjax.viz`` entry points and
isolates each plot operation in ``try/except`` so a failure in one family
(experimental, simulated, fit, residual) does not abort the others.

Public surface:
    dispatch_plots(args, config_manager, data, result) -> int

Heavy matplotlib / datashader imports are deferred to function bodies — this
keeps the plotting stack out of the import graph for non-plotting CLI
invocations.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from xpcsjax.utils.logging import _LOG_CONTEXT, get_logger, log_exception, log_once

if TYPE_CHECKING:
    import argparse

    from xpcsjax.config import ConfigManager
    from xpcsjax.optimization.nlsq.results import OptimizationResult

logger = get_logger(__name__)


# Monotonic per-call token so the per-phi ``log_once`` keys never collapse
# across independent dispatch-function calls when ``run_id`` is None (no
# configured log context). Mirrors the async_io._WAIT_ALL_CALL_COUNTER defense:
# without it, two successive calls would share a static ``"None:..."`` key in
# the process-global dedup cache and the second call's per-angle warning would
# be silently suppressed. Keeping run_id in the key still scopes by run when
# one is set; the token scopes by dispatch-function call.
_PLOT_DISPATCH_CALL_COUNTER = itertools.count()


def _current_run_id() -> str | None:
    """Read the active ``run_id`` from the log-context registry, if any.

    Used to scope ``log_once`` rate-limit keys per analysis run so a per-angle
    render failure logged once in one fit does not stay silenced for later fits
    in the same long-lived process. Returns ``None`` when no run is in context.
    """
    ctx = _LOG_CONTEXT.get() or {}
    return ctx.get("run_id")


__all__ = [
    "dispatch_plots",
    "resolve_plots_dir",
    "resolve_phi_angles_for_sim",
    "should_use_datashader",
]


# ---------------------------------------------------------------------------
# Helpers — output directory + flag resolution
# ---------------------------------------------------------------------------


def resolve_plots_dir(args: Any, config_manager: ConfigManager | None) -> Path:
    """Resolve the directory where plots will be written.

    The output ROOT is resolved by the shared
    :func:`xpcsjax.cli.config_handling.resolve_output_dir` — the same resolver
    used by result saving — so plots land under the configured output tree
    (``output.directory`` / ``output.base_directory``, or the legacy
    ``output_settings.output_dir``) rather than the process cwd. Falls back to
    the current working directory only when nothing is configured.

    A ``plots/`` subdirectory is created beneath the resolved root.
    """
    # Local import keeps the matplotlib-free CLI import graph intact and
    # avoids a circular import (commands -> plot_dispatch).
    from xpcsjax.cli.config_handling import resolve_output_dir

    root = resolve_output_dir(args, config_manager)
    if root is None:
        root = Path(".")
    plots_dir = Path(root) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def should_use_datashader(backend: str | None) -> bool:
    """Translate ``--plotting-backend`` to the ``use_datashader`` boolean.

    "auto" lets the viz layer probe optional deps; we forward True and let
    ``xpcsjax.viz.nlsq_plots`` fall back to matplotlib if Datashader is
    unavailable.
    """
    if backend in (None, "auto", "datashader"):
        return True
    return False


def resolve_phi_angles_for_sim(
    phi_angles_str: str | None,
    data: dict[str, Any] | None,
) -> np.ndarray | None:
    """Parse the ``--phi-angles`` CLI option, falling back to data angles."""
    if phi_angles_str:
        try:
            return np.array(
                [float(x.strip()) for x in phi_angles_str.split(",")],
                dtype=np.float64,
            )
        except ValueError:
            logger.warning("Could not parse --phi-angles %r; using data angles", phi_angles_str)
    if data is not None and "phi_angles_list" in data:
        return np.asarray(data["phi_angles_list"], dtype=np.float64)
    return None


# ---------------------------------------------------------------------------
# Experimental data plots (standalone QC path)
# ---------------------------------------------------------------------------


def _plot_experimental_data(data: dict[str, Any], plots_dir: Path) -> Path | None:
    """Render per-angle experimental C2 heatmaps for QC.

    Uses ``xpcsjax.viz.plot_nlsq_fit`` is unsuitable here (it requires a fit),
    so this routes through the single-panel ``plot_simulated_data`` entry
    point with the experimental array — the function plots any 2D C2 surface
    and annotates basic stats inline. One file per angle.
    """
    # Lazy: avoid pulling matplotlib into the import chain for non-plot commands.
    import matplotlib

    matplotlib.use("Agg")

    from xpcsjax.viz import plot_simulated_data

    c2_exp = np.asarray(data.get("c2_exp", data.get("c2")))
    phi_list = np.asarray(data.get("phi_angles_list", []), dtype=np.float64)
    t1 = data.get("t1")
    t2 = data.get("t2")

    if c2_exp.size == 0:
        logger.warning("No experimental c2 data to plot")
        return None

    if c2_exp.ndim == 2:
        c2_exp = c2_exp[np.newaxis, ...]

    for i in range(c2_exp.shape[0]):
        phi = float(phi_list[i]) if i < len(phi_list) else 0.0
        save_path = plots_dir / f"experimental_data_phi{int(round(phi))}.png"
        try:
            plot_simulated_data(
                c2_exp[i],
                t=np.asarray(t1) if t1 is not None else None,
                t2=np.asarray(t2) if t2 is not None else None,
                phi_deg=phi,
                save_path=save_path,
                title="Experimental C₂(t₁, t₂)",
            )
        except Exception as exc:
            log_exception(
                logger,
                exc,
                context={"operation": "plot_experimental_data", "phi": phi},
                level=logging.WARNING,
            )

    return plots_dir


# ---------------------------------------------------------------------------
# Simulated data plots (standalone, no fit required)
# ---------------------------------------------------------------------------


def _plot_simulated_from_config(
    config_manager: ConfigManager,
    contrast: float,
    offset: float,
    phi_angles_str: str | None,
    plots_dir: Path,
    data: dict[str, Any] | None,
) -> Path | None:
    """Render theoretical C2 heatmaps from the config's initial parameters.

    This evaluates the configured model at its initial-parameter vector for
    each requested phi angle and writes one PNG per angle. Useful for sanity
    checking the chosen mode + parameter ranges before running a real fit.
    """
    import matplotlib

    matplotlib.use("Agg")

    from xpcsjax.viz import plot_simulated_data

    # Per-call token so this call's per-phi log_once keys never collapse with a
    # later call's when run_id is None (process-global dedup cache).
    _call_token = next(_PLOT_DISPATCH_CALL_COUNTER)

    cfg = config_manager.get_config()
    analysis_mode = cfg.get("analysis_mode", "static_isotropic") if isinstance(cfg, dict) else None

    phi_angles = resolve_phi_angles_for_sim(phi_angles_str, data)
    if phi_angles is None or len(phi_angles) == 0:
        phi_angles = np.array([0.0, 45.0, 90.0, 135.0], dtype=np.float64)

    try:
        model = config_manager.get_model()
    except Exception as exc:
        log_exception(
            logger,
            exc,
            context={"operation": "simulated_plots_get_model"},
            level=logging.WARNING,
        )
        return None

    try:
        init_params = config_manager.get_initial_parameters()
    except Exception as exc:
        log_exception(
            logger,
            exc,
            context={"operation": "simulated_plots_get_initial_parameters"},
            level=logging.WARNING,
        )
        return None

    # Pull physical scalars from the merged config for the model call.
    cfg = config_manager.config or {}
    analyzer = cfg.get("analyzer_parameters", {}) or {}
    temporal = cfg.get("temporal", {}) or {}
    scattering = analyzer.get("scattering", {}) or {}
    q = float(scattering.get("wavevector_q", 0.01))
    geometry = analyzer.get("geometry", {}) or {}
    L = float(geometry.get("stator_rotor_gap", 1.0))
    dt = float(analyzer.get("dt", temporal.get("dt", 1.0)))

    # Evaluate the model on its configured *elapsed-time* grid, not on raw
    # frame-index axes. Mirrors heterodyne.core.HeterodyneModel.from_config:
    # t = arange(n_times) * dt + t_start with t_start = dt (the first usable
    # frame sits at 1×dt). The two-component cross term is
    # cos(q·cos φ·∫v(t')dt'); feeding frame indices (or a bare arange that
    # ignores dt and the frame window) into that integral collapses the
    # fringe structure and yields a qualitatively wrong C2 surface.
    if "start_frame" in analyzer and "end_frame" in analyzer:
        n_times = int(analyzer["end_frame"]) - int(analyzer["start_frame"]) + 1
        t_start = dt
    else:
        n_times = int(temporal.get("time_length", 1000))
        t_start = float(temporal.get("t_start", dt))
    t_model = np.arange(n_times, dtype=np.float64) * dt + t_start

    # Display extent: prefer the experiment's true elapsed-time axis when it is
    # present and shape-compatible; otherwise fall back to the model grid.
    t_extent = t_model
    if data is not None:
        t_disp = data.get("t1_original", data.get("t1"))
        if t_disp is not None and len(np.asarray(t_disp)) == n_times:
            t_extent = np.asarray(t_disp, dtype=np.float64)

    t1_arr = t_model
    t2_arr = t_model

    # Order the dict-form init params per the active-parameter list.
    try:
        active = config_manager.get_active_parameters()
        params_arr = np.array([float(init_params[name]) for name in active], dtype=np.float64)
    except Exception as exc:
        log_exception(
            logger,
            exc,
            context={"operation": "simulated_plots_order_init_params"},
            level=logging.WARNING,
        )
        return None

    for phi in phi_angles:
        try:
            c2_sim = _evaluate_model_c2(
                model,
                params_arr,
                float(phi),
                t1_arr,
                t2_arr,
                q=q,
                L=L,
                contrast=contrast,
                offset=offset,
                dt=dt,
            )
        except Exception as exc:
            run_id = _current_run_id()
            log_once(
                logger,
                logging.WARNING,
                f"{run_id}:{_call_token}:plot_render_fail:simulated_evaluate_model_c2",
                "Could not evaluate model c2 at phi=%s: %s (further per-angle failures suppressed)",
                phi,
                exc,
            )
            continue

        # _evaluate_model_c2 already returns a fully-scaled c2 surface
        # (compute_g2 applies offset + contrast*g1^2 internally; the
        # heterodyne compute_g1 path returns c2 directly). Do NOT re-scale
        # here — that produced a double-scaling artifact.
        c2_surface = np.asarray(c2_sim)

        save_path = plots_dir / f"simulated_c2_phi{int(round(float(phi)))}.png"
        try:
            plot_simulated_data(
                c2_surface,
                t=t_extent,
                t2=t_extent,
                phi_deg=float(phi),
                contrast=contrast,
                offset=offset,
                analysis_mode=analysis_mode,
                save_path=save_path,
            )
        except Exception as exc:
            run_id = _current_run_id()
            log_once(
                logger,
                logging.WARNING,
                f"{run_id}:{_call_token}:plot_render_fail:simulated_plot",
                "Failed to render simulated plot for phi=%s: %s (further "
                "per-angle failures suppressed)",
                phi,
                exc,
            )

    return plots_dir


def _evaluate_model_c2(
    model: Any,
    params: np.ndarray,
    phi_deg: float,
    t1: np.ndarray,
    t2: np.ndarray,
    *,
    q: float,
    L: float,
    contrast: float,
    offset: float,
    dt: float,
) -> np.ndarray:
    """Evaluate a model's fully-scaled c2 surface at a single phi angle.

    Two model families with different evaluation surfaces:

    * ``CombinedModel`` (modes static_anisotropic / static_isotropic /
      laminar_flow) exposes
      ``compute_g2(params, t1, t2, phi, q, L, contrast, offset, dt)`` which
      applies ``c2 = offset + contrast*g1^2`` internally.
    * ``HeterodyneModel`` (mode two_component) has NO ``compute_g2``; its
      ``compute_g1(params, t1, t2, phi, q, L, dt)`` already returns the full
      ``c2`` surface (contrast/offset are baked into the 14-element param
      vector), per that method's own docstring.

    Either way the return is a fully-scaled c2 surface — the caller must
    NOT re-apply contrast/offset.

    Parity note: this mirrors ``xpcsjax.viz.nlsq_plots`` model dispatch.
    ``CombinedModel.compute_g2`` applies ``offset + contrast*g1^2``
    internally, but ``HeterodyneModel.compute_g1`` calls the kernel with
    ``contrast=1.0, offset=0.0`` (heterodyne_model.py:151), returning a
    NORMALIZED surface — so this branch must apply the scaling itself, the
    same way ``viz.nlsq_plots`` does at its HeterodyneModel branch.
    """
    import jax.numpy as jnp

    phi_arr = jnp.asarray([phi_deg], dtype=jnp.float64)
    p_arr = jnp.asarray(params, dtype=jnp.float64)
    t1_j = jnp.asarray(t1, dtype=jnp.float64)
    t2_j = jnp.asarray(t2, dtype=jnp.float64)

    g2_method = getattr(model, "compute_g2", None)
    if g2_method is not None:
        # CombinedModel path — compute_g2 applies offset + contrast*g1^2.
        out = g2_method(p_arr, t1_j, t2_j, phi_arr, q, L, contrast, offset, dt)
        arr = np.asarray(out)
    else:
        # HeterodyneModel path — compute_g1 returns a normalized surface
        # (kernel called with contrast=1/offset=0). Apply scaling here to
        # match the viz layer's HeterodyneModel branch.
        g1_method = getattr(model, "compute_g1", None)
        if g1_method is None:
            raise AttributeError(f"{type(model).__name__} has neither compute_g2 nor compute_g1")
        g1_sq = np.asarray(g1_method(p_arr, t1_j, t2_j, phi_arr, q, L, dt))
        arr = offset + contrast * g1_sq
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


# ---------------------------------------------------------------------------
# Post-fit plots (NLSQ result available)
# ---------------------------------------------------------------------------


def _generate_post_fit_plots(
    args: Any,
    config_manager: ConfigManager,
    data: dict[str, Any],
    result: OptimizationResult,
    plots_dir: Path,
) -> Path | None:
    """Generate the full 3-panel / residual / simulated artifact set.

    Delegates to ``xpcsjax.viz.generate_nlsq_plots``, which is the high-level
    orchestrator and handles per-angle dispatch, datashader fallback, and
    artifact serialization (NPZ + JSON) under ``output_dir``.
    """
    import matplotlib

    matplotlib.use("Agg")

    from xpcsjax.viz import generate_nlsq_plots

    # Write the full artifact set under the ``plots/`` directory so the post-fit
    # dispatch lands in the same place as every other plot path
    # (``_plot_experimental_data``, ``_save_simulated_only``,
    # ``_save_fit_comparison_only``) and matches the "Plots written to <root>/plots"
    # message logged by ``dispatch_plots``. ``generate_nlsq_plots`` creates its
    # own ``simulated_data/`` subdirectory beneath whatever output_dir it is given,
    # so the fitted artifacts land at ``<root>/plots/simulated_data/``. The main
    # NLSQ results (nlsq_result.json/.npz) are written separately to ``<root>`` by
    # ``save_results`` and are intentionally not nested under ``plots/``.

    use_datashader = should_use_datashader(getattr(args, "plotting_backend", "auto"))
    parallel = bool(getattr(args, "parallel_plots", False))

    try:
        model = config_manager.get_model()
    except Exception as exc:
        log_exception(
            logger,
            exc,
            context={"operation": "post_fit_plots_get_model"},
            level=logging.WARNING,
        )
        return None

    cfg = config_manager.get_config()

    try:
        generate_nlsq_plots(
            model=model,
            result=result,
            data=data,
            config=cfg,
            output_dir=plots_dir,
            use_datashader=use_datashader,
            parallel=parallel,
        )
    except Exception as exc:
        log_exception(
            logger,
            exc,
            context={"operation": "generate_nlsq_plots"},
            level=logging.WARNING,
        )
        return None

    return plots_dir


def _save_fit_comparison_only(
    config_manager: ConfigManager,
    data: dict[str, Any],
    result: OptimizationResult,
    plots_dir: Path,
) -> Path | None:
    """Lightweight ``--save-plots`` path: per-angle fit + residual figures only.

    Used when the user wants fit-vs-experiment comparisons saved but doesn't
    want the full datashader / artifact dump that ``generate_nlsq_plots``
    produces.
    """
    import matplotlib

    matplotlib.use("Agg")

    from xpcsjax.viz import plot_nlsq_fit, plot_residual_map

    # Per-call token so this call's per-phi log_once keys never collapse with a
    # later call's when run_id is None (process-global dedup cache).
    _call_token = next(_PLOT_DISPATCH_CALL_COUNTER)

    try:
        model = config_manager.get_model()
    except Exception as exc:
        log_exception(
            logger,
            exc,
            context={"operation": "fit_comparison_get_model"},
            level=logging.WARNING,
        )
        return None

    c2_exp = np.asarray(data.get("c2_exp", data.get("c2")))
    phi_list = np.asarray(data.get("phi_angles_list", []), dtype=np.float64)
    t1 = data.get("t1")
    t2 = data.get("t2")

    # Pull physical scalars for compute_g2 — same source as the simulated path.
    cfg = config_manager.config or {}
    _analyzer = cfg.get("analyzer_parameters", {}) or {}
    _scattering = _analyzer.get("scattering", {}) or {}
    _geometry = _analyzer.get("geometry", {}) or {}
    _q = float(_scattering.get("wavevector_q", 0.01))
    _L = float(_geometry.get("stator_rotor_gap", 1.0))
    _dt = float(_analyzer.get("dt", 1.0))
    # Pull scaling from the fitted result when available, else defaults.
    _contrast = float(getattr(result, "contrast", 0.3) or 0.3)
    _offset = float(getattr(result, "offset", 1.0) or 1.0)

    if c2_exp.size == 0 or len(phi_list) == 0:
        logger.warning("Missing c2_exp or phi_angles_list; skipping fit-comparison plots")
        return None

    if c2_exp.ndim == 2:
        c2_exp = c2_exp[np.newaxis, ...]

    t1_arr = np.asarray(t1, dtype=np.float64) if t1 is not None else None
    t2_arr = np.asarray(t2, dtype=np.float64) if t2 is not None else None

    for i, phi in enumerate(phi_list):
        if i >= c2_exp.shape[0]:
            break
        try:
            c2_fit = _evaluate_model_c2(
                model,
                np.asarray(result.parameters),
                float(phi),
                t1_arr if t1_arr is not None else np.arange(c2_exp.shape[1], dtype=np.float64),
                t2_arr if t2_arr is not None else np.arange(c2_exp.shape[2], dtype=np.float64),
                q=_q,
                L=_L,
                contrast=_contrast,
                offset=_offset,
                dt=_dt,
            )
        except Exception as exc:
            run_id = _current_run_id()
            log_once(
                logger,
                logging.WARNING,
                f"{run_id}:{_call_token}:plot_render_fail:fit_comparison_evaluate_c2",
                "Could not evaluate fitted c2 at phi=%s: %s (further per-angle "
                "failures suppressed)",
                phi,
                exc,
            )
            continue

        suffix = f"_phi{int(round(float(phi)))}"
        try:
            plot_nlsq_fit(
                c2_exp[i],
                np.asarray(c2_fit),
                t=t1_arr,
                t2=t2_arr,
                phi_deg=float(phi),
                reduced_chi_squared=result.reduced_chi_squared,
                save_path=plots_dir / f"nlsq_fit{suffix}.png",
            )
        except Exception as exc:
            run_id = _current_run_id()
            log_once(
                logger,
                logging.WARNING,
                f"{run_id}:{_call_token}:plot_render_fail:fit_comparison_nlsq_fit",
                "plot_nlsq_fit failed for phi=%s: %s (further per-angle failures suppressed)",
                phi,
                exc,
            )

        try:
            plot_residual_map(
                c2_exp[i],
                np.asarray(c2_fit),
                t=t1_arr,
                t2=t2_arr,
                phi_deg=float(phi),
                save_path=plots_dir / f"nlsq_residuals{suffix}.png",
            )
        except Exception as exc:
            run_id = _current_run_id()
            log_once(
                logger,
                logging.WARNING,
                f"{run_id}:{_call_token}:plot_render_fail:fit_comparison_residual_map",
                "plot_residual_map failed for phi=%s: %s (further per-angle failures suppressed)",
                phi,
                exc,
            )

    return plots_dir


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def dispatch_plots(
    args: argparse.Namespace | Any,
    config_manager: ConfigManager | None,
    data: dict[str, Any] | None,
    result: OptimizationResult | None,
) -> int:
    """Fan out CLI plotting flags to the right ``xpcsjax.viz`` entry points.

    Routing rules:
        * ``--plot-experimental-data`` → ``_plot_experimental_data`` (standalone)
        * ``--plot-simulated-data``    → ``_plot_simulated_from_config`` (standalone)
        * ``--save-plots`` + result    → ``_save_fit_comparison_only``
        * ``args.plot`` + result       → full ``generate_nlsq_plots`` artifact dump

    Each operation is independently isolated so a failure in one family
    does not abort the others.

    Parameters
    ----------
    args
        Parsed CLI namespace.
    config_manager
        The active ``ConfigManager`` (may be ``None`` for the simplest paths).
    data
        Loaded XPCS data dict (``c2_exp``, ``t1``, ``t2``, ``phi_angles_list``)
        or ``None`` when only synthetic plots were requested.
    result
        The NLSQ optimization result, or ``None`` when no fit was performed.

    Returns
    -------
    int
        ``0`` on success — by convention, individual plot failures are logged
        but do not produce a non-zero exit code. Returns ``0`` even when no
        plots are produced (caller decides whether that is an error).
    """
    plots_dir = resolve_plots_dir(args, config_manager)

    plot_exp = bool(getattr(args, "plot_experimental_data", False))
    plot_sim = bool(getattr(args, "plot_simulated_data", False))
    save_plots = bool(getattr(args, "save_plots", False))
    plot_after_fit = bool(getattr(args, "plot", True))

    # Each plot helper returns the directory it actually wrote into (or None
    # when it wrote nothing). We log the *actual* set of written locations
    # rather than the pre-computed ``plots_dir`` so the "Plots written to …"
    # message can never drift from where files really landed — the failure mode
    # that previously had post-fit artifacts scattered into the output root
    # while the log claimed ``<root>/plots``.
    written: set[Path] = set()

    def _record(out: Path | None) -> None:
        if out is not None:
            written.add(Path(out))

    # ---- Standalone QC paths (no fit needed) ----
    if plot_exp:
        if data is not None:
            try:
                _record(_plot_experimental_data(data, plots_dir))
            except Exception as exc:
                log_exception(
                    logger,
                    exc,
                    context={"operation": "dispatch_experimental_data"},
                    level=logging.WARNING,
                )
        else:
            logger.warning("--plot-experimental-data requested but no data was loaded")

    if plot_sim:
        if config_manager is not None:
            try:
                contrast = float(getattr(args, "contrast", 0.3))
                offset = float(getattr(args, "offset_sim", 1.0))
                phi_str = getattr(args, "phi_angles", None)
                _record(
                    _plot_simulated_from_config(
                        config_manager, contrast, offset, phi_str, plots_dir, data
                    )
                )
            except Exception as exc:
                log_exception(
                    logger,
                    exc,
                    context={"operation": "dispatch_simulated_data"},
                    level=logging.WARNING,
                )
        else:
            logger.warning("--plot-simulated-data requested but no config_manager")

    # ---- Post-fit paths (require result + config_manager + data) ----
    if result is not None and config_manager is not None and data is not None:
        if save_plots:
            try:
                _record(_save_fit_comparison_only(config_manager, data, result, plots_dir))
            except Exception as exc:
                log_exception(
                    logger,
                    exc,
                    context={"operation": "dispatch_fit_comparison"},
                    level=logging.WARNING,
                )

        if plot_after_fit and not (plot_exp or plot_sim):
            # Full artifact dump path — only when the user did NOT explicitly
            # request a standalone plot mode (those skip the fit entirely).
            try:
                _record(_generate_post_fit_plots(args, config_manager, data, result, plots_dir))
            except Exception as exc:
                log_exception(
                    logger,
                    exc,
                    context={"operation": "dispatch_post_fit"},
                    level=logging.WARNING,
                )

    if written:
        logger.info("Plots written to %s", ", ".join(sorted(str(p) for p in written)))
    else:
        logger.debug("dispatch_plots: nothing to do (no flags set or required inputs missing)")

    return 0
