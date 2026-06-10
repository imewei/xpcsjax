"""Optimization execution for xpcsjax CLI.

Manages NLSQ fitting runs. NLSQ is the only optimizer pathway in xpcsjax;
Bayesian sampling is out of scope (see project CLAUDE.md).

Public surface
--------------
``run_nlsq(args, config_manager, data) -> OptimizationResult``
    Execute an NLSQ fit and return the aggregate result. CLI flags
    (``--multistart``, ``--multistart-n``, ``--max-iterations``,
    ``--tolerance``, ``--verbose`` …) are mapped onto the
    :class:`ConfigManager` before dispatch.

The dispatch itself is owned by :func:`xpcsjax.fit_nlsq`: it inspects
the merged config's ``analysis_mode`` and routes ``two_component``
(heterodyne) to the multi-phi heterodyne path, all other modes to
:func:`fit_nlsq_jax`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from xpcsjax import OptimizationResult, fit_nlsq
from xpcsjax.io.nlsq_writers import save_nlsq_json_files
from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    from xpcsjax.config import ConfigManager

logger = get_logger(__name__)


__all__ = [
    "run_nlsq",
    "apply_cli_overrides",
    "format_nlsq_summary",
]


# ---------------------------------------------------------------------------
# CLI -> config translation
# ---------------------------------------------------------------------------


_NLSQ_SECTION = ("optimization", "nlsq")


def _set_nested(cfg: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Set ``cfg[path[0]][path[1]]...`` = value, creating dicts as needed."""
    node = cfg
    for key in path[:-1]:
        existing = node.get(key)
        if not isinstance(existing, dict):
            existing = {}
            node[key] = existing
        node = existing
    node[path[-1]] = value


def apply_cli_overrides(
    args: argparse.Namespace,
    config_manager: ConfigManager,
) -> None:
    """Merge ``args`` flags into ``config_manager.config`` in place.

    Only flags this module is responsible for (the NLSQ runtime knobs in
    the task spec) are written here. Other flags — ``--mode``, ``--phi``,
    ``--output`` — are handled in earlier CLI stages.
    """
    cfg = config_manager.config
    if not isinstance(cfg, dict):
        return

    multistart = getattr(args, "multistart", None)
    if multistart is not None:
        _set_nested(cfg, (*_NLSQ_SECTION, "multi_start", "enable"), bool(multistart))
        logger.info("CLI override: multi_start.enable = %s", bool(multistart))

    multistart_n = getattr(args, "multistart_n", None)
    if multistart_n is not None:
        _set_nested(cfg, (*_NLSQ_SECTION, "multi_start", "n_starts"), int(multistart_n))
        logger.info("CLI override: multi_start.n_starts = %d", int(multistart_n))

    max_iterations = getattr(args, "max_iterations", None)
    if max_iterations is not None:
        _set_nested(cfg, (*_NLSQ_SECTION, "max_iterations"), int(max_iterations))
        logger.info("CLI override: nlsq.max_iterations = %d", int(max_iterations))

    tolerance = getattr(args, "tolerance", None)
    if tolerance is not None:
        ftol = float(tolerance)
        _set_nested(cfg, (*_NLSQ_SECTION, "ftol"), ftol)
        _set_nested(cfg, (*_NLSQ_SECTION, "xtol"), ftol)
        logger.info("CLI override: nlsq.ftol = nlsq.xtol = %g", ftol)

    verbose = bool(getattr(args, "verbose", False))
    quiet = bool(getattr(args, "quiet", False))
    if verbose or quiet:
        # 0 = silent, 1 = default, 2 = chatty
        v = 0 if quiet else (2 if verbose else 1)
        _set_nested(cfg, (*_NLSQ_SECTION, "verbose"), v)


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


def format_nlsq_summary(result: OptimizationResult) -> str:
    """Return a short human-readable summary of an OptimizationResult."""
    lines = [
        f"  status:           {result.convergence_status}",
        f"  iterations:       {result.iterations}",
        f"  chi^2:            {result.chi_squared:.6g}",
        f"  reduced chi^2:    {result.reduced_chi_squared:.6g}",
        f"  quality:          {result.quality_flag}",
        f"  wall time (s):    {result.execution_time:.3f}",
    ]
    if result.recovery_actions:
        lines.append(f"  recovery actions: {result.recovery_actions}")
    return "\n".join(lines)


def _warn_nlsq_bound_saturation(result: OptimizationResult) -> None:
    """Warn for parameters with zero/near-zero uncertainty.

    Mirrors the upstream heterodyne diagnostic. Bound saturation here is
    informational only — xpcsjax has no Bayesian downstream consumer.
    """
    if result.uncertainties is None:
        return

    try:
        from xpcsjax.config.parameter_registry import DEFAULT_REGISTRY  # type: ignore

        registry: Any = DEFAULT_REGISTRY
    except ImportError:
        registry = None

    param_names: list[str] | None = None
    diagnostics = result.nlsq_diagnostics or {}
    if isinstance(diagnostics, dict):
        names = diagnostics.get("parameter_names")
        if isinstance(names, (list, tuple)):
            param_names = [str(n) for n in names]

    values = np.asarray(result.parameters).ravel()
    uncertainties = np.asarray(result.uncertainties).ravel()
    if values.size != uncertainties.size:
        return

    saturated: list[str] = []
    for i, unc in enumerate(uncertainties):
        if float(unc) >= 1e-30:
            continue
        name = param_names[i] if param_names and i < len(param_names) else f"param[{i}]"
        val = float(values[i])
        hint = ""
        if registry is not None:
            try:
                info = registry[name]
                if abs(val - info.min_bound) < 1e-10 * max(abs(info.min_bound), 1.0):
                    hint = " [AT LOWER BOUND]"
                elif abs(val - info.max_bound) < 1e-10 * max(abs(info.max_bound), 1.0):
                    hint = " [AT UPPER BOUND]"
                else:
                    hint = " [DEGENERATE JACOBIAN]"
            except (KeyError, AttributeError):
                pass
        logger.warning("NLSQ bound saturation: %s = %.4g +/- 0%s", name, val, hint)
        saturated.append(name)

    if saturated:
        logger.warning(
            "%d parameter(s) saturated at bounds or degenerate: %s",
            len(saturated),
            saturated,
        )


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------


def _build_param_dict(result: OptimizationResult) -> dict[str, Any]:
    """Pack (parameters, uncertainties) into the JSON shape writers expect."""
    values = np.asarray(result.parameters).ravel()
    uncerts = (
        np.asarray(result.uncertainties).ravel()
        if result.uncertainties is not None
        else np.full(values.shape, np.nan)
    )
    names: list[str] = []
    diagnostics = result.nlsq_diagnostics or {}
    if isinstance(diagnostics, dict):
        cand = diagnostics.get("parameter_names")
        if isinstance(cand, (list, tuple)):
            names = [str(n) for n in cand]
    if len(names) != values.size:
        names = [f"param_{i}" for i in range(values.size)]

    return {
        name: {
            "value": float(values[i]),
            "uncertainty": (
                float(uncerts[i]) if i < uncerts.size and np.isfinite(uncerts[i]) else None
            ),
        }
        for i, name in enumerate(names)
    }


def _build_analysis_dict(
    result: OptimizationResult,
    config_manager: ConfigManager,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Top-level analysis metadata for ``analysis_results_nlsq.json``."""
    mode = ""
    if hasattr(config_manager, "config") and isinstance(config_manager.config, dict):
        mode = str(config_manager.config.get("analysis_mode", ""))

    c2 = data.get("c2_exp", data.get("c2"))
    dataset_info: dict[str, Any] = {}
    if c2 is not None:
        arr = np.asarray(c2)
        dataset_info = {
            "shape": list(arr.shape),
            "n_points": int(arr.size),
        }

    return {
        "method": "nlsq",
        "analysis_mode": mode,
        "fit_quality": {
            "chi_squared": float(result.chi_squared),
            "reduced_chi_squared": float(result.reduced_chi_squared),
            "quality_flag": result.quality_flag,
        },
        "dataset_info": dataset_info,
    }


def _build_convergence_dict(result: OptimizationResult) -> dict[str, Any]:
    """Convergence metrics block for ``convergence_metrics.json``."""
    return {
        "status": result.convergence_status,
        "iterations": int(result.iterations),
        "execution_time_s": float(result.execution_time),
        "recovery_actions": list(result.recovery_actions or []),
        "device_info": result.device_info or {},
        "sigma_is_default": bool(result.sigma_is_default),
    }


def _save_results(
    result: OptimizationResult,
    config_manager: ConfigManager,
    data: dict[str, Any],
    output_dir: Path,
) -> None:
    """Persist parameters, analysis summary, and convergence diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        save_nlsq_json_files(
            _build_param_dict(result),
            _build_analysis_dict(result, config_manager, data),
            _build_convergence_dict(result),
            output_dir,
        )
        logger.info("Saved NLSQ JSON results -> %s", output_dir)
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Could not write NLSQ JSON outputs: %s", exc)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _resolve_output_dir(args: argparse.Namespace) -> Path | None:
    """Return the ``--output`` directory if specified, else None."""
    out = getattr(args, "output", None)
    if out is None:
        return None
    return Path(out)


def run_nlsq(
    args: argparse.Namespace,
    config_manager: ConfigManager,
    data: dict[str, Any],
) -> OptimizationResult:
    """Execute the NLSQ fit and return the result.

    Parameters
    ----------
    args
        Parsed CLI arguments (from :func:`xpcsjax.cli.args_parser.create_parser`).
    config_manager
        Already-merged :class:`ConfigManager`. Mode is set; this function
        only layers CLI overrides on top.
    data
        XPCS data dict accepted by :func:`xpcsjax.fit_nlsq`. Keys depend on
        mode: homodyne uses ``phi_angles_list`` / ``c2_exp`` / ``t1`` / ``t2``,
        heterodyne uses ``c2_exp`` (or ``c2``) and ``phi_angles_list``
        (or ``phi_angles`` / ``phi``).

    Returns
    -------
    OptimizationResult
        The aggregate fit result. For ``two_component`` (heterodyne) mode
        the per-angle scaling lives under ``result.nlsq_diagnostics``.
    """
    logger.info("Starting NLSQ analysis")

    apply_cli_overrides(args, config_manager)

    mode = ""
    if hasattr(config_manager, "config") and isinstance(config_manager.config, dict):
        mode = str(config_manager.config.get("analysis_mode", ""))
    logger.info("Analysis mode: %s", mode or "<unset>")

    if getattr(args, "no_jit", False):
        logger.info("JAX_DISABLE_JIT=1 (set in main bootstrap); fit will run uncompiled")

    # Dispatch through the public gateway. ``fit_nlsq`` routes
    # ``two_component`` -> heterodyne multi-phi path, otherwise -> fit_nlsq_jax.
    try:
        result = fit_nlsq(data, config_manager)
    except Exception:
        logger.exception("NLSQ fit raised an exception")
        raise

    if not isinstance(result, OptimizationResult):
        # MultiStartResult or other wrappers expose ``.best`` (OptimizationResult).
        best = getattr(result, "best", None)
        if isinstance(best, OptimizationResult):
            result = best
        else:
            raise TypeError(
                f"fit_nlsq returned unexpected type {type(result).__name__}; "
                "expected OptimizationResult"
            )
    assert isinstance(result, OptimizationResult)  # Pyright narrowing

    _warn_nlsq_bound_saturation(result)

    logger.info(
        "NLSQ Results\n%s\n%s\n%s",
        "=" * 50,
        format_nlsq_summary(result),
        "=" * 50,
    )

    if not result.success:
        logger.warning(
            "NLSQ did not converge (status=%s). Consider --multistart, "
            "tighter --tolerance, or revising bounds in the YAML config.",
            result.convergence_status,
        )

    output_dir = _resolve_output_dir(args)
    if output_dir is not None:
        _save_results(result, config_manager, data, output_dir)

    logger.info("NLSQ analysis complete")
    return result
