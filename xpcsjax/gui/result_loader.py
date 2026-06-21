"""Read a persisted NLSQ result directory into a lightweight summary.

JAX-free: stdlib only. The GUI process loads this in-process to display a fit's
outcome without touching the worker-side numerical stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_RESULT_JSON = "nlsq_result.json"


@dataclass(frozen=True)
class ResultSummary:
    """A display-ready summary of a completed fit (parsed from nlsq_result.json)."""

    result_dir: Path
    success: bool
    convergence_status: str
    chi_squared: float | None
    reduced_chi_squared: float | None
    quality_flag: str
    parameters: dict[str, float]
    # Per-parameter uncertainty (None where the fit reported none). Defaulted so
    # existing direct ResultSummary(...) constructions (F/G tests) stay valid.
    uncertainties: dict[str, float | None] = field(default_factory=dict)
    # NLSQ diagnostics block (anti-degeneracy layer activations, etc.). Defaulted so
    # existing direct ResultSummary(...) constructions stay valid.
    diagnostics: dict = field(default_factory=dict)


def _as_float(value: object) -> float | None:
    """Coerce a JSON number to float, or None for null/non-numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def load_result_summary(result_dir: str | Path) -> ResultSummary | None:
    """Load ``<result_dir>/nlsq_result.json`` into a :class:`ResultSummary`.

    Returns ``None`` (never raises) when the file is absent or unreadable, so the
    GUI can degrade gracefully on a partial or missing result.
    """
    result_dir = Path(result_dir)
    json_path = result_dir / _RESULT_JSON
    if not json_path.is_file():
        return None
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    meta_raw = payload.get("metadata")
    meta = meta_raw if isinstance(meta_raw, dict) else {}
    params_raw = payload.get("parameters")
    params_blob = params_raw if isinstance(params_raw, dict) else {}
    parameters = {
        str(name): _as_float(info.get("value"))
        for name, info in params_blob.items()
        if isinstance(info, dict) and _as_float(info.get("value")) is not None
    }
    uncertainties = {
        str(name): _as_float(info.get("uncertainty"))
        for name, info in params_blob.items()
        if isinstance(info, dict) and _as_float(info.get("value")) is not None
    }

    # Coerce a non-dict (e.g. null) nlsq_diagnostics to {} — symmetric with the
    # metadata/parameters guards above — so a partial/external result JSON never
    # yields a non-dict diagnostics that crashes the inspector on iteration.
    diag_raw = meta.get("nlsq_diagnostics", {})
    diagnostics = diag_raw if isinstance(diag_raw, dict) else {}
    return ResultSummary(
        result_dir=result_dir,
        success=bool(meta.get("success", False)),
        convergence_status=str(meta.get("convergence_status", "unknown")),
        chi_squared=_as_float(meta.get("chi_squared")),
        reduced_chi_squared=_as_float(meta.get("reduced_chi_squared")),
        quality_flag=str(meta.get("quality_flag", "")),
        parameters={k: v for k, v in parameters.items() if v is not None},
        uncertainties=uncertainties,
        diagnostics=diagnostics,
    )
