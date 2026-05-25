"""Top-level command dispatcher for the xpcsjax CLI.

NLSQ-only by design: there is exactly one fitting path (``run_nlsq``),
plus two standalone plot-only modes (``--plot-experimental-data`` and
``--plot-simulated-data``) that skip optimization entirely.

The dispatcher's contract:
    * Parse-time validation has already happened in ``main.py``.
    * This function returns an int exit code:
        - 0 on success / converged optimization
        - 2 on optimizer non-convergence (still writes outputs)
        - non-zero raised by callees on hard errors (propagates to main)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from xpcsjax.cli.config_handling import load_and_merge_config, resolve_output_dir
from xpcsjax.cli.data_pipeline import load_and_validate_data, resolve_phi_angles
from xpcsjax.cli.optimization_runner import run_nlsq
from xpcsjax.cli.plot_dispatch import dispatch_plots
from xpcsjax.cli.result_saving import save_results
from xpcsjax.utils.logging import get_logger, log_exception

if TYPE_CHECKING:
    from xpcsjax.config.manager import ConfigManager
    from xpcsjax.optimization.nlsq.results import OptimizationResult

logger = get_logger(__name__)


def dispatch_command(args: argparse.Namespace) -> int:
    """Run the command implied by ``args``.

    Branches:
        * Standalone plot modes (``--plot-experimental-data`` /
          ``--plot-simulated-data``): skip optimization, render plots, return 0.
        * Default: load config + data → run NLSQ → save → plot.

    Returns:
        Exit code (0 ok, 2 non-convergence, callee exceptions bubble up).
    """
    cfg_manager = load_and_merge_config(args.config, args)

    standalone_plot = bool(
        getattr(args, "plot_experimental_data", False)
        or getattr(args, "plot_simulated_data", False)
    )

    if standalone_plot:
        return _dispatch_standalone_plot(args, cfg_manager)

    return _dispatch_fit(args, cfg_manager)


# ---------------------------------------------------------------------------
# Standalone plot modes (skip optimization)
# ---------------------------------------------------------------------------


def _dispatch_standalone_plot(
    args: argparse.Namespace,
    cfg_manager: ConfigManager,
) -> int:
    """Plot experimental data or simulated C2 heatmaps without optimizing."""
    if getattr(args, "plot_simulated_data", False):
        logger.info("Standalone mode: plot simulated C2 heatmaps from config parameters")
        # Simulated mode doesn't need experimental data on disk
        return dispatch_plots(args, cfg_manager, data=None, result=None)

    # plot_experimental_data — needs data
    logger.info("Standalone mode: plot experimental data for QC")
    data = load_and_validate_data(args, cfg_manager)
    return dispatch_plots(args, cfg_manager, data=data, result=None)


# ---------------------------------------------------------------------------
# Default: NLSQ fit pipeline
# ---------------------------------------------------------------------------


def _dispatch_fit(
    args: argparse.Namespace,
    cfg_manager: ConfigManager,
) -> int:
    """Load data → run NLSQ → save → plot. Returns 0 / 2."""
    # Load + filter experimental data
    try:
        data = load_and_validate_data(args, cfg_manager)
    except Exception as exc:
        log_exception(logger, exc, context={"command": "load_data"})
        raise

    phi_angles = resolve_phi_angles(args, cfg_manager)
    if phi_angles is not None:
        logger.info("Analyzing %d phi angle(s): %s", len(phi_angles), phi_angles)

    # Run NLSQ
    try:
        result: OptimizationResult = run_nlsq(args, cfg_manager, data)
    except Exception as exc:
        log_exception(logger, exc, context={"command": "run_nlsq"})
        raise

    # Persist results
    output_dir = _resolve_output_dir(args, cfg_manager)
    if output_dir is not None:
        try:
            save_results(
                result,
                output_dir=output_dir,
                output_format=args.output_format,
                config_manager=cfg_manager,
                args=args,
            )
        except Exception as exc:
            log_exception(logger, exc, context={"command": "save_results"})
            raise

    # Plotting
    plot_enabled = bool(getattr(args, "plot", True))
    if plot_enabled or getattr(args, "save_plots", False):
        try:
            dispatch_plots(args, cfg_manager, data=data, result=result)
        except Exception as exc:
            # Don't fail the whole run on plotting errors — log and continue
            log_exception(logger, exc, context={"command": "dispatch_plots"})

    # Exit code: 0 on convergence, 2 otherwise
    success = bool(getattr(result, "success", False))
    if not success:
        logger.warning(
            "NLSQ optimizer did NOT converge — outputs were written but the "
            "fit is not trustworthy. Review nlsq_diagnostics in the result."
        )
        return 2
    return 0


def _resolve_output_dir(
    args: argparse.Namespace,
    cfg_manager: ConfigManager,
) -> Path | None:
    """Resolve the effective output directory (CLI > YAML > None).

    Thin wrapper over :func:`config_handling.resolve_output_dir`, the single
    source of truth shared with the plot path so result and plot artifacts
    always land under the same configured root.
    """
    return resolve_output_dir(args, cfg_manager)


__all__ = ["dispatch_command"]
