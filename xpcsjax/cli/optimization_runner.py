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

from xpcsjax import OptimizationResult
from xpcsjax.cli.config_handling import resolve_output_dir
from xpcsjax.io.nlsq_writers import save_nlsq_json_files
from xpcsjax.service.fit import FitOverrides, apply_overrides, run_fit
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


def _overrides_from_args(args: argparse.Namespace) -> FitOverrides:
    """Build typed :class:`FitOverrides` from parsed CLI args."""
    return FitOverrides(
        multistart=getattr(args, "multistart", None),
        multistart_n=getattr(args, "multistart_n", None),
        max_iterations=getattr(args, "max_iterations", None),
        tolerance=getattr(args, "tolerance", None),
        verbose=bool(getattr(args, "verbose", False)),
        quiet=bool(getattr(args, "quiet", False)),
    )


def apply_cli_overrides(
    args: argparse.Namespace,
    config_manager: ConfigManager,
) -> None:
    """Merge CLI flags into ``config_manager.config`` (delegates to the service)."""
    apply_overrides(config_manager, _overrides_from_args(args))


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
        # NaN/inf uncertainty means no covariance solve was run (e.g. a global
        # escape: CMA-ES/multistart returns np.full(n, np.nan)) -- not bound
        # saturation. `nan >= 1e-30` is False, so without this guard it would
        # fall through and be misreported as "+/- 0".
        if not np.isfinite(unc):
            continue
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
            except (KeyError, AttributeError) as e:
                logger.debug("Could not resolve bound-saturation hint for %s: %s", name, e)
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

    mode = ""
    if isinstance(getattr(config_manager, "config", None), dict):
        mode = str(config_manager.config.get("analysis_mode", ""))
    logger.info("Analysis mode: %s", mode or "<unset>")

    if getattr(args, "no_jit", False):
        logger.info("JAX_DISABLE_JIT=1 (set in main bootstrap); fit will run uncompiled")

    # Service owns override-application + dispatch + result normalization;
    # this adapter keeps the CLI-flavored side effects below.
    try:
        result = run_fit(config_manager, data, overrides=_overrides_from_args(args))
    except Exception:
        logger.exception("NLSQ fit raised an exception")
        raise

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

    # Writer 1 of 2 (intentional dual-format output — audit [23], confirmed
    # not a clobber): this emits the homodyne-compatible trio
    # parameters.json / analysis_results_nlsq.json / convergence_metrics.json
    # for downstream homodyne tooling. The CLI dispatcher (commands._dispatch_fit
    # -> result_saving.save_results) separately writes the native
    # nlsq_result.json/.npz. Distinct filenames in the same directory; keep both.
    output_dir = resolve_output_dir(args, config_manager)
    if output_dir is not None:
        _save_results(result, config_manager, data, output_dir)

    logger.info("NLSQ analysis complete")
    return result
