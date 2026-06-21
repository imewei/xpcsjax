"""Simulated-data plot family for the xpcsjax CLI.

Renders theoretical C2 heatmaps from the config's initial parameters
(standalone — no fit required). The only module in the CLI plot split that
holds a direct ``import jax.numpy as jnp``; that import is function-local
inside ``_evaluate_model_c2`` so this module does NOT load JAX at import time.

Heavy matplotlib imports are likewise deferred to function bodies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from xpcsjax.cli.plot_backend import _PLOT_DISPATCH_CALL_COUNTER, _current_run_id
from xpcsjax.utils.logging import get_logger, log_exception, log_once

if TYPE_CHECKING:
    from xpcsjax.config import ConfigManager

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers — phi-angle parsing for simulated data
# ---------------------------------------------------------------------------


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
    # frame-index axes. Mirrors heterodyne.HeterodyneModel.from_config:
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
