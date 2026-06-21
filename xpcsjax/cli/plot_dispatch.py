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

from xpcsjax.cli.plot_backend import (
    resolve_plots_dir,
    should_use_datashader,
)
from xpcsjax.cli.plot_families.experimental import _plot_experimental_data
from xpcsjax.cli.plot_families.postfit import (
    _generate_post_fit_plots,
    _save_fit_comparison_only,
)
from xpcsjax.cli.plot_families.simulated import (
    _plot_simulated_from_config,
    resolve_phi_angles_for_sim,
)
from xpcsjax.utils.logging import get_logger, log_exception

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
