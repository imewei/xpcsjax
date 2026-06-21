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

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from xpcsjax.cli.plot_backend import (
    _PLOT_DISPATCH_CALL_COUNTER,
    _current_run_id,
    resolve_plots_dir,
    should_use_datashader,
)
from xpcsjax.cli.plot_families.experimental import _plot_experimental_data
from xpcsjax.cli.plot_families.simulated import (
    _plot_simulated_from_config,
    resolve_phi_angles_for_sim,
)
from xpcsjax.utils.logging import get_logger, log_exception, log_once

if TYPE_CHECKING:
    import argparse

    from xpcsjax.config import ConfigManager
    from xpcsjax.optimization.nlsq.results import OptimizationResult

logger = get_logger(__name__)


__all__ = [
    "dispatch_plots",
    "resolve_plots_dir",
    "resolve_phi_angles_for_sim",
    "should_use_datashader",
]


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
    """Generate the full 3-panel / residual / simulated artifact set (delegates)."""
    from xpcsjax.service.plots import generate_plots

    return generate_plots(
        result,
        data,
        config_manager,
        plots_dir,
        use_datashader=should_use_datashader(getattr(args, "plotting_backend", "auto")),
        parallel=bool(getattr(args, "parallel_plots", False)),
    )


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

    # The fitted c2 surface is rendered per-angle via the shared
    # ``_evaluate_c2_per_angle`` extractor, which resolves the model-specific
    # parameter layout (homodyne scaling-first vs heterodyne physics-first) and
    # the per-angle contrast/offset from ``result``/diagnostics. q/L/dt are read
    # from ``cfg`` inside that helper.
    cfg = config_manager.config or {}

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
            # Use the model-aware shared extractor: it splits result.parameters
            # into per-angle (contrast, offset) + physical block rather than
            # passing the whole scaling-prefixed vector as physics with defaults.
            from xpcsjax.viz.nlsq_plots import _evaluate_c2_per_angle

            c2_fit = _evaluate_c2_per_angle(model, result, data, cfg, float(phi))
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
