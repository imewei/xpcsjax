"""Result persistence utilities for xpcsjax CLI.

Writes :class:`OptimizationResult` instances to disk as JSON, NPZ, or both.
xpcsjax is NLSQ-only by design; there is no posterior / MCMC code path here.
"""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    from argparse import Namespace

    from xpcsjax import ConfigManager, OptimizationResult

__all__ = [
    "save_results",
    "save_results_json",
    "save_results_npz",
]

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# JSON-serialization helpers
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Recursively coerce a value into JSON-serializable primitives.

    Handles numpy scalars / arrays, Paths, datetimes, and nested
    dict/list/tuple structures. Anything else falls back to ``str(value)``.

    Non-finite floats (NaN / +-inf), which arise from diverged fits, are
    coerced to ``None`` — ``json.dumps`` would otherwise emit bare
    ``NaN`` / ``Infinity`` tokens that are not valid JSON and break strict
    downstream parsers.
    """
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        fval = float(value)
        return fval if math.isfinite(fval) else None
    if isinstance(value, np.ndarray):
        # Replace non-finite entries with None to keep the JSON valid.
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (Path, datetime.datetime, datetime.date)):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


# ---------------------------------------------------------------------------
# Structured result extraction
# ---------------------------------------------------------------------------


def _extract_parameters(
    result: OptimizationResult,
    parameter_names: list[str] | None,
) -> dict[str, dict[str, float | None]]:
    """Map each parameter name to its value and (optional) uncertainty.

    If ``parameter_names`` is missing or its length does not match the
    parameter vector, falls back to ``param_0, param_1, ...`` indexing so
    we never raise during result persistence.
    """
    params = np.asarray(result.parameters)
    n = params.shape[0]

    if parameter_names is None or len(parameter_names) != n:
        names = [f"param_{i}" for i in range(n)]
    else:
        names = list(parameter_names)

    uncertainties: np.ndarray | None = (
        np.asarray(result.uncertainties) if result.uncertainties is not None else None
    )

    out: dict[str, dict[str, float | None]] = {}
    for i, name in enumerate(names):
        unc = float(uncertainties[i]) if uncertainties is not None else None
        out[name] = {"value": float(params[i]), "uncertainty": unc}
    return out


def _extract_metadata(result: OptimizationResult) -> dict[str, Any]:
    """Flatten NLSQ fit-quality metrics into a JSON-friendly dict."""
    meta: dict[str, Any] = {
        "success": bool(result.success),
        "convergence_status": result.convergence_status,
        "message": result.message,
        "iterations": int(result.iterations),
        "chi_squared": float(result.chi_squared),
        "reduced_chi_squared": float(result.reduced_chi_squared),
        "execution_time": float(result.execution_time),
        "quality_flag": result.quality_flag,
        "sigma_is_default": bool(result.sigma_is_default),
    }
    if result.recovery_actions:
        meta["recovery_actions"] = list(result.recovery_actions)
    if result.device_info:
        meta["device_info"] = _json_safe(result.device_info)
    if result.nlsq_diagnostics:
        meta["nlsq_diagnostics"] = _json_safe(result.nlsq_diagnostics)
    if result.streaming_diagnostics:
        meta["streaming_diagnostics"] = _json_safe(result.streaming_diagnostics)
    if result.stratification_diagnostics is not None:
        # StratificationDiagnostics is a dataclass; surface its public fields.
        diag = result.stratification_diagnostics
        if hasattr(diag, "__dict__"):
            meta["stratification_diagnostics"] = _json_safe(vars(diag))
        else:
            meta["stratification_diagnostics"] = str(diag)
    return meta


def _config_summary(config_manager: ConfigManager | None) -> dict[str, Any]:
    """Extract a small, JSON-safe header describing the run configuration."""
    if config_manager is None:
        return {}

    summary: dict[str, Any] = {}
    for attr in ("mode", "data_type", "config_path"):
        if hasattr(config_manager, attr):
            summary[attr] = _json_safe(getattr(config_manager, attr))
    # Parameter names are needed downstream for labeled output. xpcsjax's
    # ConfigManager exposes get_active_parameters() (not the heterodyne-style
    # get_parameter_names()).
    if hasattr(config_manager, "get_active_parameters"):
        try:
            summary["parameter_names"] = list(config_manager.get_active_parameters())
        except Exception:  # pragma: no cover - defensive: never fail save on config introspection
            logger.warning(
                "ConfigManager.get_active_parameters() failed; saved result will "
                "omit parameter_names (downstream parsers may need them).",
                exc_info=True,
            )
    return summary


def _resolve_parameter_names(config_manager: ConfigManager | None) -> list[str] | None:
    """Pull parameter names from the ConfigManager when available."""
    if config_manager is None:
        return None
    if hasattr(config_manager, "get_active_parameters"):
        try:
            return list(config_manager.get_active_parameters())
        except Exception:
            logger.warning(
                "Could not resolve parameter names from ConfigManager; output will be unlabeled.",
                exc_info=True,
            )
    return None


# ---------------------------------------------------------------------------
# Format-specific writers
# ---------------------------------------------------------------------------


def save_results_json(
    result: OptimizationResult,
    output_dir: Path,
    config_manager: ConfigManager | None = None,
    args: Namespace | None = None,
    *,
    filename: str = "nlsq_result.json",
) -> Path:
    """Write the optimization summary (no residuals) to a JSON file.

    Residual / covariance arrays are intentionally omitted here — they
    live in the NPZ companion file. JSON stays human-readable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_names = _resolve_parameter_names(config_manager)

    payload: dict[str, Any] = {
        "schema": "xpcsjax.nlsq.result/v1",
        "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "config": _config_summary(config_manager),
        "parameters": _extract_parameters(result, parameter_names),
        "metadata": _extract_metadata(result),
    }
    if args is not None:
        payload["cli_args"] = _json_safe(vars(args))

    path = output_dir / filename
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    logger.info("Saved NLSQ result JSON to %s", path)
    return path


def save_results_npz(
    result: OptimizationResult,
    output_dir: Path,
    config_manager: ConfigManager | None = None,
    *,
    filename: str = "nlsq_result.npz",
    residuals: np.ndarray | None = None,
) -> Path:
    """Write parameter arrays, residuals, and fit metadata to a single NPZ.

    The NPZ is the full-fidelity artifact: float64 arrays preserved
    exactly, suitable for downstream re-analysis.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_names = _resolve_parameter_names(config_manager)

    arrays: dict[str, np.ndarray] = {
        "parameters": np.asarray(result.parameters, dtype=np.float64),
        "uncertainties": np.asarray(result.uncertainties, dtype=np.float64),
        "covariance": np.asarray(result.covariance, dtype=np.float64),
        "chi_squared": np.asarray(result.chi_squared, dtype=np.float64),
        "reduced_chi_squared": np.asarray(result.reduced_chi_squared, dtype=np.float64),
        "iterations": np.asarray(result.iterations, dtype=np.int64),
        "execution_time": np.asarray(result.execution_time, dtype=np.float64),
    }

    if parameter_names is not None:
        arrays["parameter_names"] = np.array(parameter_names, dtype=object)
    if residuals is not None:
        arrays["residuals"] = np.asarray(residuals, dtype=np.float64)

    metadata_blob = json.dumps(_json_safe(_extract_metadata(result)))
    arrays["metadata_json"] = np.array(metadata_blob, dtype=object)
    arrays["config_json"] = np.array(
        json.dumps(_json_safe(_config_summary(config_manager))), dtype=object
    )

    path = output_dir / filename
    np.savez(path, **arrays)  # type: ignore[arg-type]  # numpy stub: **kwargs ArrayLike
    logger.info("Saved NLSQ result NPZ to %s", path)
    return path


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def save_results(
    result: OptimizationResult,
    output_dir: Path,
    output_format: str,
    config_manager: ConfigManager | None,
    args: Namespace | None,
) -> None:
    """Persist an :class:`OptimizationResult` in the requested format(s).

    Args:
        result: The completed NLSQ optimization result.
        output_dir: Destination directory. Created if missing.
        output_format: One of ``"json"``, ``"npz"``, or ``"both"``.
        config_manager: Optional ConfigManager whose mode, data_type, and
            parameter names are recorded alongside the result.
        args: Optional parsed CLI namespace. When present its attributes
            are serialized into the JSON output for provenance.

    Raises:
        ValueError: If ``output_format`` is not one of the accepted values.
    """
    fmt = output_format.lower().strip()
    if fmt not in {"json", "npz", "both"}:
        raise ValueError(
            f"Unknown output_format {output_format!r}; expected 'json', 'npz', or 'both'."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    residuals = getattr(args, "residuals", None) if args is not None else None

    if fmt in ("json", "both"):
        save_results_json(result, output_dir, config_manager, args)
    if fmt in ("npz", "both"):
        save_results_npz(result, output_dir, config_manager, residuals=residuals)

    logger.info(
        "save_results complete: format=%s, dir=%s, status=%s",
        fmt,
        output_dir,
        result.convergence_status,
    )
