"""Post-fit plot helpers for the xpcsjax CLI.

Contains the two functions that require an ``OptimizationResult``:

* ``_generate_post_fit_plots`` — full 3-panel / residual / simulated artifact
  set, delegated to ``xpcsjax.service.plots.generate_plots``.
* ``_save_fit_comparison_only`` — lightweight per-angle fit + residual figures
  for ``--save-plots``.

Heavy imports (``matplotlib``, ``xpcsjax.viz``, ``xpcsjax.service.plots``) are
deferred to function bodies to keep the JAX-free startup invariant.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from xpcsjax.cli.plot_backend import (
    _PLOT_DISPATCH_CALL_COUNTER,
    _current_run_id,
    should_use_datashader,
)
from xpcsjax.utils.logging import get_logger, log_exception, log_once

if TYPE_CHECKING:
    from xpcsjax.config import ConfigManager
    from xpcsjax.optimization.nlsq.results import OptimizationResult

logger = get_logger(__name__)


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

            c2_fit = _evaluate_c2_per_angle(model, result, data, cfg, float(phi), phi_index=i)
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

        # Mirrors viz/nlsq_plots.py naming: bare int(round(phi)) collides.
        suffix = f"_phi_{i:03d}_{float(phi):.3f}deg"
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
