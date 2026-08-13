"""Result persistence service (relocated from xpcsjax.cli.result_saving).

Writes :class:`OptimizationResult` instances to disk as JSON, NPZ, or both.
xpcsjax is NLSQ-only by design; there is no posterior / MCMC code path here.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from xpcsjax.utils.logging import get_logger
from xpcsjax.utils.path_validation import get_safe_output_dir

if TYPE_CHECKING:
    from argparse import Namespace

    from xpcsjax import ConfigManager, OptimizationResult

__all__ = [
    "merge_fitted_c2",
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
        # A 0-D array (e.g. ``np.array(3.5)``) is a scalar: ``tolist()`` returns a
        # bare Python scalar, not a list, so route it back through the scalar
        # branches above (finite-float -> None coercion included) instead of
        # iterating a non-iterable.
        if value.ndim == 0:
            return _json_safe(value.item())
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
    params = np.asarray(result.parameters).ravel()
    n = params.size

    if parameter_names is None or len(parameter_names) != n:
        if parameter_names is not None:
            logger.warning(
                "parameter_names length (%d) does not match parameter vector "
                "length (%d); falling back to generic param_0, param_1, ... "
                "labels in the saved result.",
                len(parameter_names),
                n,
            )
        names = [f"param_{i}" for i in range(n)]
    else:
        names = list(parameter_names)

    # Flatten and use uncertainties ONLY when the count matches the parameter
    # vector exactly. ``__post_init__`` lets a 0-D scalar (size==1, ndim==0) and
    # other off-shape arrays through; ``.ravel()`` normalizes 0-D -> (1,) so
    # ``unc[i]`` never raises IndexError, and the length-match gate drops any
    # array that does not correspond 1:1 to the parameters.
    uncertainties: np.ndarray | None = None
    if result.uncertainties is not None:
        unc_flat = np.asarray(result.uncertainties).ravel()
        if unc_flat.size == n:
            uncertainties = unc_flat

    out: dict[str, dict[str, float | None]] = {}
    for i, name in enumerate(names):
        unc = float(uncertainties[i]) if uncertainties is not None else None
        out[name] = {"value": float(params[i]), "uncertainty": unc}
    return out


def _shaped_or_nan(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Coerce *value* to a float64 array of *shape*, or an all-NaN array.

    ``__post_init__`` admits ``None`` and 0-D/empty placeholders for any
    parameter count; stored verbatim they break readers that index the NPZ 1:1
    against ``parameters``.
    """
    if value is not None:
        arr = np.asarray(value, dtype=np.float64)
        if arr.size == int(np.prod(shape, dtype=int)):
            return arr.reshape(shape)
        logger.warning(
            "Value with size %d does not match expected shape %s (size %d); "
            "saving an all-NaN placeholder instead.",
            arr.size,
            shape,
            int(np.prod(shape, dtype=int)),
        )
    return np.full(shape, np.nan, dtype=np.float64)


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


def _config_summary(
    config_manager: ConfigManager | None,
    parameter_names: list[str] | None = None,
) -> dict[str, Any]:
    """Extract a small, JSON-safe header describing the run configuration.

    ``parameter_names``, when given, should be the SAME list already resolved
    by :func:`_resolve_parameter_names` for this result -- otherwise this
    header would carry a shorter, physics-only list) that silently disagrees
    with the (scaling + physics) keys in the ``parameters`` block next to it.
    """
    if config_manager is None:
        return {}

    summary: dict[str, Any] = {}
    # xpcsjax's ConfigManager exposes the mode via config["analysis_mode"] (the
    # raw value, read directly to avoid the .analysis_mode property's deferred
    # ValueError), the path via .config_file, and data_type from the config
    # dict — NOT via mode/data_type/config_path attributes (heterodyne-style).
    config = getattr(config_manager, "config", None)
    if isinstance(config, dict):
        summary["mode"] = _json_safe(config.get("analysis_mode"))
        data_type = config.get("data_type")
        if data_type is None:
            exp = config.get("experimental_data")
            if isinstance(exp, dict):
                data_type = exp.get("data_type")
        summary["data_type"] = _json_safe(data_type)
    summary["config_path"] = _json_safe(getattr(config_manager, "config_file", None))
    if parameter_names is not None:
        summary["parameter_names"] = list(parameter_names)
    return summary


def _synthesize_scaling_names(physics_names: list[str], n_total: int) -> list[str] | None:
    """Reconstruct (scaling + physics) names from the vector length alone.

    Homodyne fits prepend a per-angle scaling head ahead of the physics tail,
    laid out as ``[contrast_0..N, offset_0..N, <physics>]`` -- a block, not
    interleaved, layout (see ``optimization/nlsq/core.py``, which slices
    ``result.parameters[:n_angles]`` / ``[n_angles:2*n_angles]`` for exactly
    this reason). Config alone doesn't carry which per-angle mode a given fit
    used (constant / averaged / individual), but the length delta against the
    physics-only count pins down the scaling-head size unambiguously:
    ``n_total == n_physics + 2*n_angles``.

    Returns ``None`` when the delta isn't a valid non-negative even
    scaling-head size, so the caller falls back to generic ``param_i`` labels
    instead of guessing.
    """
    extra = n_total - len(physics_names)
    if extra == 0:
        return list(physics_names)
    if extra < 0 or extra % 2 != 0:
        return None
    n_angles = extra // 2
    scaling = [f"contrast_{i}" for i in range(n_angles)] + [f"offset_{i}" for i in range(n_angles)]
    return [*scaling, *physics_names]


def _resolve_parameter_names(
    config_manager: ConfigManager | None,
    result: OptimizationResult | None = None,
) -> list[str] | None:
    """Pull parameter names for labeling the fitted parameter vector.

    Three tiers, most-authoritative first:

    1. ``result.nlsq_diagnostics["parameter_names"]`` -- the optimizer's own
       label list, in the exact order of ``result.parameters``. Only the
       heterodyne (``two_component``) code paths populate this key today.
    2. ``ConfigManager.get_active_parameters()`` (physics-only by design),
       extended with synthesized ``contrast_i``/``offset_i`` names sized to
       match ``result.parameters`` via :func:`_synthesize_scaling_names` --
       this is what actually covers homodyne fits (static_isotropic,
       static_anisotropic, laminar_flow), since none of them populate tier 1.
    3. Bare physics-only names, if a vector was never provided to size the
       scaling head against.
    """
    if result is not None:
        diagnostics = result.nlsq_diagnostics or {}
        if isinstance(diagnostics, dict):
            cand = diagnostics.get("parameter_names")
            if isinstance(cand, (list, tuple)) and cand:
                # Tier 1 is the optimizer's own label list, authoritative and
                # already guaranteed to match result.parameters by
                # construction -- returned as-is, no length re-check against
                # result.parameters (some callers pass a lightweight/mocked
                # result whose .parameters was never meant to be touched here).
                return [str(n) for n in cand]
            if diagnostics:
                logger.debug(
                    "result.nlsq_diagnostics present but has no usable "
                    "'parameter_names' list; falling back to ConfigManager-derived names.",
                )
    if config_manager is None or not hasattr(config_manager, "get_active_parameters"):
        return None
    try:
        physics_names = list(config_manager.get_active_parameters())
    except Exception:
        logger.warning(
            "Could not resolve parameter names from ConfigManager; output will be unlabeled.",
            exc_info=True,
        )
        return None

    if result is None:
        return physics_names

    n_total = np.asarray(result.parameters).size
    synthesized = _synthesize_scaling_names(physics_names, n_total)
    names = synthesized if synthesized is not None else physics_names

    # Single length-vs-vector validation point for tier 2 (was previously
    # duplicated: _extract_parameters checked it independently while
    # _config_summary did not check it at all, letting a mismatched list
    # reach the JSON header while the parameters block silently fell back
    # to generic names).
    if len(names) != n_total:
        logger.warning(
            "Resolved parameter_names length (%d) does not match parameter "
            "vector length (%d); output will be unlabeled.",
            len(names),
            n_total,
        )
        return None
    return names


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

    Residual / covariance arrays are intentionally omitted here -- they live
    in the NPZ companion file. JSON stays human-readable.

    Parameters
    ----------
    result : OptimizationResult
        The completed NLSQ optimization result.
    output_dir : pathlib.Path
        Destination directory. Created if missing.
    config_manager : ConfigManager or None, optional
        Source of mode / data_type / parameter names for the JSON header.
    args : argparse.Namespace or None, optional
        Parsed CLI namespace; when present, serialized under ``cli_args`` for
        provenance.
    filename : str, keyword-only, optional
        Output file name within *output_dir*.

    Returns
    -------
    pathlib.Path
        Path to the written JSON file.
    """
    output_dir = get_safe_output_dir(output_dir)
    parameter_names = _resolve_parameter_names(config_manager, result)

    payload: dict[str, Any] = {
        "schema": "xpcsjax.nlsq.result/v1",
        "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "config": _config_summary(config_manager, parameter_names),
        "parameters": _extract_parameters(result, parameter_names),
        "metadata": _extract_metadata(result),
    }
    if args is not None:
        # Exclude ``residuals`` (and any other array-shaped attribute a
        # future caller might set on the namespace): this file intentionally
        # omits residual/covariance arrays -- they live in the NPZ companion
        # -- so a blind vars(args) dump must not reintroduce them via cli_args.
        cli_args = {k: v for k, v in vars(args).items() if k != "residuals"}
        payload["cli_args"] = _json_safe(cli_args)

    path = output_dir / filename
    tmp_path = output_dir / (filename + ".tmp")
    tmp_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
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

    The NPZ is the full-fidelity artifact: float64 arrays preserved exactly,
    suitable for downstream re-analysis.

    Parameters
    ----------
    result : OptimizationResult
        The completed NLSQ optimization result.
    output_dir : pathlib.Path
        Destination directory. Created if missing.
    config_manager : ConfigManager or None, optional
        Source of parameter names recorded alongside the arrays.
    filename : str, keyword-only, optional
        Output file name within *output_dir*.
    residuals : numpy.ndarray or None, keyword-only, optional
        Optional residual array stored under the ``residuals`` key.

    Returns
    -------
    pathlib.Path
        Path to the written NPZ file.
    """
    output_dir = get_safe_output_dir(output_dir)
    parameter_names = _resolve_parameter_names(config_manager, result)

    params = np.asarray(result.parameters, dtype=np.float64)
    n_params = params.size
    uncertainties = _shaped_or_nan(result.uncertainties, (n_params,))
    covariance = _shaped_or_nan(result.covariance, (n_params, n_params))

    arrays: dict[str, np.ndarray] = {
        "parameters": params,
        "uncertainties": uncertainties,
        "covariance": covariance,
        "chi_squared": np.asarray(result.chi_squared, dtype=np.float64),
        "reduced_chi_squared": np.asarray(result.reduced_chi_squared, dtype=np.float64),
        "iterations": np.asarray(result.iterations, dtype=np.int64),
        "execution_time": np.asarray(result.execution_time, dtype=np.float64),
    }

    if parameter_names is not None:
        # No dtype=object: a fixed-width unicode array (numpy infers '<U...')
        # holds the same strings without forcing readers to pass
        # allow_pickle=True, matching the SEC-1 no-pickle convention this
        # project already enforces in data/xpcs_loader.py and
        # data/performance_engine.py.
        arrays["parameter_names"] = np.array(parameter_names)
    if residuals is not None:
        arrays["residuals"] = np.asarray(residuals, dtype=np.float64)

    metadata_blob = json.dumps(_json_safe(_extract_metadata(result)))
    arrays["metadata_json"] = np.array(metadata_blob)
    arrays["config_json"] = np.array(
        json.dumps(_json_safe(_config_summary(config_manager, parameter_names)))
    )

    path = output_dir / filename
    # Atomic write: np.savez writes to a temp file first, then os.replace moves it
    # into place so a reader (e.g. GUI loading on Finished) never sees a partial file.
    # np.savez appends ".npz" when the path doesn't already end with it, so we use
    # a NamedTemporaryFile with suffix=".npz" (already has the extension) to ensure
    # the temp name itself is a valid .npz path.
    with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".npz", delete=False) as tmp_fh:
        tmp_name = tmp_fh.name
    try:
        np.savez(tmp_name, **arrays)  # type: ignore[arg-type]  # numpy stub: **kwargs ArrayLike
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
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

    Parameters
    ----------
    result : OptimizationResult
        The completed NLSQ optimization result.
    output_dir : pathlib.Path
        Destination directory. Created if missing.
    output_format : str
        One of ``"json"``, ``"npz"``, or ``"both"`` (case-insensitive).
    config_manager : ConfigManager or None
        Optional manager whose mode, data_type, and parameter names are
        recorded alongside the result.
    args : argparse.Namespace or None
        Optional parsed CLI namespace. When present, its attributes are
        serialized into the JSON output for provenance, and ``args.residuals``
        (if any) is stored in the NPZ.

    Raises
    ------
    ValueError
        If ``output_format`` is not one of the accepted values.
    """
    fmt = output_format.lower().strip()
    if fmt not in {"json", "npz", "both"}:
        raise ValueError(
            f"Unknown output_format {output_format!r}; expected 'json', 'npz', or 'both'."
        )

    output_dir = get_safe_output_dir(output_dir)

    residuals = getattr(args, "residuals", None) if args is not None else None

    # For "both", write the durable full-fidelity NPZ FIRST so a failure in the
    # human-readable JSON serialization (e.g. an off-shape field) cannot discard
    # the numeric artifact after an expensive run.
    if fmt in ("npz", "both"):
        save_results_npz(result, output_dir, config_manager, residuals=residuals)
    if fmt in ("json", "both"):
        save_results_json(result, output_dir, config_manager, args)

    logger.info(
        "save_results complete: format=%s, dir=%s, status=%s",
        fmt,
        output_dir,
        result.convergence_status,
    )


# (source key in c2_fitted_data.npz, destination key in nlsq_result.npz)
# pairs pulled by merge_fitted_c2. Deliberately excludes that file's own
# "params" / "contrast" / "offset" / "reduced_chi_squared" -- those duplicate
# what save_results_npz already writes under different (and authoritative)
# names, so merging them would create confusing near-duplicate keys. "q" is
# renamed to "wavevector_q" on merge since save_results_npz never writes a
# wavevector value under any name, so there is no collision to avoid.
_FITTED_C2_MERGE_KEYS = (
    ("c2_exp", "c2_exp"),
    ("c2_fitted", "c2_fitted"),
    ("residuals", "residuals"),
    ("t1", "t1"),
    ("t2", "t2"),
    ("phi_angles", "phi_angles"),
    ("q", "wavevector_q"),
)


def merge_fitted_c2(npz_path: Path, fitted_c2_npz: Path) -> bool:
    """Merge experimental/fitted/residual c2 arrays into the primary result NPZ.

    ``save_results_npz`` writes only scalars, parameters, and covariance --
    the raw ``(n_phi, n_t1, n_t2)`` correlation surfaces are never included,
    so a saved result has no way to reconstruct what was actually fit without
    re-running the pipeline. When post-fit plotting runs, the "simulated"
    plot family (part of :func:`xpcsjax.viz.generate_nlsq_plots`'s default
    set) already computes and writes exactly this data to
    ``<plots_dir>/simulated_data/c2_fitted_data.npz``. This merges those
    arrays into the primary NPZ too, so downstream re-analysis doesn't need
    to dig into the plots directory for them.

    Best-effort: this only enriches an already-saved result with convenience
    arrays that were computed elsewhere, so it never raises. Returns
    ``False`` (with a WARNING logged) if either file is missing or the merge
    fails for any reason -- the primary result on disk is left untouched in
    that case.

    Parameters
    ----------
    npz_path : pathlib.Path
        The primary result NPZ written by :func:`save_results_npz` (or
        :func:`save_results` with ``output_format`` including ``"npz"``).
    fitted_c2_npz : pathlib.Path
        The ``c2_fitted_data.npz`` written by the "simulated" plot family.
        Callers resolve this as
        ``<plots_dir>/simulated_data/c2_fitted_data.npz``, matching the
        layout ``xpcsjax.viz.nlsq_plots`` writes.

    Returns
    -------
    bool
        ``True`` if the merge happened, ``False`` otherwise (missing input,
        or the merge/write failed).
    """
    if not fitted_c2_npz.exists():
        logger.debug(
            "No fitted-c2 NPZ at %s (plotting likely skipped or failed); "
            "primary result NPZ will not carry c2_exp/c2_fitted/residuals/wavevector_q.",
            fitted_c2_npz,
        )
        return False
    if not npz_path.exists():
        logger.debug("No primary result NPZ at %s to merge fitted c2 into.", npz_path)
        return False

    # Freshness guard: save_results_npz always writes npz_path BEFORE
    # plotting runs, so a fitted_c2_npz genuinely produced by THIS run is
    # never older than npz_path. If plotting silently failed this run (the
    # CLI dispatcher swallows per-family plot errors and proceeds regardless),
    # a stale fitted_c2_npz left over from a PREVIOUS run in the same output
    # directory would otherwise get merged in here -- pairing this run's
    # parameters/chi_squared with a previous run's c2 arrays with no error
    # raised anywhere. Reject anything strictly older than npz_path instead.
    try:
        fitted_mtime = fitted_c2_npz.stat().st_mtime
        npz_mtime = npz_path.stat().st_mtime
    except OSError as exc:
        logger.warning("Could not stat NPZ files while merging fitted c2 arrays: %s", exc)
        return False
    if fitted_mtime < npz_mtime:
        logger.warning(
            "Fitted-c2 NPZ %s is older than %s (stale leftover from a previous "
            "run in this output directory?); skipping merge to avoid pairing "
            "this run's parameters with another run's c2 arrays.",
            fitted_c2_npz,
            npz_path,
        )
        return False

    try:
        with np.load(npz_path, allow_pickle=False) as existing:
            arrays: dict[str, np.ndarray] = dict(existing.items())
        with np.load(fitted_c2_npz, allow_pickle=False) as fitted:
            for src_key, dest_key in _FITTED_C2_MERGE_KEYS:
                if src_key not in fitted.files:
                    continue
                if dest_key in arrays:
                    logger.debug(
                        "Skipping merge of %r into %s: key already present.", dest_key, npz_path
                    )
                    continue
                arrays[dest_key] = fitted[src_key]
    except Exception as exc:
        # Broad catch is deliberate: this function's contract is "never
        # raises" (best-effort enrichment of an already-saved result), so a
        # corrupted/truncated NPZ (zipfile.BadZipFile, etc.) must be handled
        # the same as any other read failure, not just OSError/ValueError.
        logger.warning("Could not read NPZ files while merging fitted c2 arrays: %s", exc)
        return False

    # np.savez appends ".npz" when the given name doesn't already end with
    # it (mirrors save_results_npz's own comment on this exact footgun) --
    # NamedTemporaryFile with suffix=".npz" keeps the temp name and the
    # actually-written name identical, so os.replace below targets the
    # right file instead of silently no-op'ing on a missing source.
    with tempfile.NamedTemporaryFile(dir=npz_path.parent, suffix=".npz", delete=False) as tmp_fh:
        tmp_path = Path(tmp_fh.name)
    try:
        np.savez(tmp_path, **arrays)  # type: ignore[arg-type]  # numpy stub: **kwargs ArrayLike
        os.replace(tmp_path, npz_path)
    except Exception as exc:
        logger.warning("Could not write merged NPZ %s: %s", npz_path, exc)
        try:
            os.unlink(tmp_path)
        except OSError:
            # Best-effort cleanup of the temp file; already reporting the
            # write failure above, so a missing/unremovable tmp file here
            # is not itself an error worth surfacing.
            pass
        return False

    logger.info("Merged experimental/fitted/residual c2 arrays into %s", npz_path)
    return True
