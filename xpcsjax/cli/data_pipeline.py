"""Data loading and validation pipeline for xpcsjax CLI.

Ported from heterodyne/cli/data_pipeline.py. Adapted to xpcsjax's
NLSQ-only surface (no CMC/MCMC paths), xpcsjax's analysis-mode taxonomy
(``static_anisotropic`` / ``static_isotropic`` / ``laminar_flow`` /
``two_component``), and the dict-shaped return of
:func:`xpcsjax.load_xpcs_data`.

Public API:
    * :func:`load_and_validate_data` -- load XPCS data, apply phi filtering.
    * :func:`resolve_phi_angles`     -- pick phi angles from CLI or config.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

import numpy as np

from xpcsjax import ConfigManager, load_xpcs_data
from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    # Re-import for type-only narrowing; runtime import already done above.
    from xpcsjax.config.manager import ConfigManager as _ConfigManager  # noqa: F401

logger = get_logger(__name__)

# Common azimuthal angles used in XPCS experiments (degrees).
COMMON_XPCS_ANGLES: list[int] = [0, 30, 45, 60, 90, 120, 135, 150, 180]

# Tolerant key spellings emitted by load_xpcs_data / XPCSDataLoader.
_C2_KEYS: tuple[str, ...] = ("c2_exp", "c2")
_PHI_KEYS: tuple[str, ...] = ("phi_angles_list", "phi_angles", "phi")
_T1_KEYS: tuple[str, ...] = ("t1",)
_T2_KEYS: tuple[str, ...] = ("t2",)


def _pick(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first present key's value, else None."""
    for k in keys:
        if k in data and data[k] is not None:
            return data[k]
    return None


def _norm_scalar(a: float) -> float:
    """Normalize a single angle to [-180, 180]."""
    v = a % 360.0
    return v - 360.0 if v > 180.0 else v


def _apply_phi_filtering(
    data_phi_angles: np.ndarray,
    phi_cfg: dict[str, Any],
) -> list[float] | None:
    """Apply ``phi_filtering`` config to select angles from data.

    Returns matched data angles as a list, or ``None`` if no match.
    """
    target_ranges = phi_cfg.get("target_ranges", [])
    if not target_ranges:
        return None

    arr = np.asarray(data_phi_angles, dtype=float)
    normalized: np.ndarray = np.where(
        (arr % 360) > 180, (arr % 360) - 360, arr % 360
    )
    tol = float(phi_cfg.get("tolerance", 5.0))
    selected_mask = np.zeros(len(normalized), dtype=bool)

    for rng in target_ranges:
        if isinstance(rng, dict):
            lo = _norm_scalar(float(rng.get("min_angle", -10.0)))
            hi = _norm_scalar(float(rng.get("max_angle", 10.0)))
        elif isinstance(rng, (list, tuple)) and len(rng) == 2:
            lo = _norm_scalar(float(rng[0]))
            hi = _norm_scalar(float(rng[1]))
        elif isinstance(rng, (int, float)):
            center = _norm_scalar(float(rng))
            lo, hi = center - tol, center + tol
        else:
            continue

        if lo <= hi:
            selected_mask |= (normalized >= lo) & (normalized <= hi)
        else:
            # Wrap-around range (e.g. [170, -170]).
            selected_mask |= (normalized >= lo) | (normalized <= hi)

    if not np.any(selected_mask):
        return None

    return [float(a) for a in arr[selected_mask]]


def load_and_validate_data(
    args: argparse.Namespace,
    config_manager: ConfigManager,
) -> dict[str, Any]:
    """Load XPCS experimental data and apply phi-angle filtering.

    The return is the raw dict produced by :func:`xpcsjax.load_xpcs_data`,
    augmented with a ``"phi_angles_selected"`` key holding the
    post-filter angle list used by the rest of the pipeline.

    Args:
        args: Parsed CLI arguments (may carry ``--phi`` overrides).
        config_manager: Validated :class:`ConfigManager`.

    Returns:
        Dict with keys including (subject to loader version):
        ``c2_exp`` / ``c2``, ``phi_angles_list`` / ``phi_angles`` / ``phi``,
        ``t1``, ``t2``, plus ``"phi_angles_selected"``.

    Raises:
        ValueError: If essential keys are missing from the loader output.
    """
    cfg = config_manager.get_config()
    analysis_mode = cfg.get("analysis_mode", "<unknown>")
    data_type = cfg.get("data_type", cfg.get("experimental_data", {}).get("data_type"))
    if data_type not in (None, "aps_old", "aps_u"):
        logger.warning(
            "Unrecognized data_type=%r in config (expected 'aps_old' or 'aps_u')",
            data_type,
        )

    logger.info(
        "Loading XPCS data (analysis_mode=%s, data_type=%s)",
        analysis_mode,
        data_type,
    )

    # xpcsjax.load_xpcs_data accepts the merged config dict directly.
    data: dict[str, Any] = load_xpcs_data(config_dict=cfg)

    c2 = _pick(data, _C2_KEYS)
    if c2 is None:
        raise ValueError(
            "load_xpcs_data returned no correlation matrix; "
            f"expected one of {_C2_KEYS}"
        )
    data_phi = _pick(data, _PHI_KEYS)
    t1 = _pick(data, _T1_KEYS)
    t2 = _pick(data, _T2_KEYS)

    c2_arr = np.asarray(c2)
    logger.info(
        "Loaded XPCS data: c2 shape=%s, %d phi angles, t1=%s, t2=%s",
        c2_arr.shape,
        0 if data_phi is None else len(np.asarray(data_phi).ravel()),
        None if t1 is None else np.asarray(t1).shape,
        None if t2 is None else np.asarray(t2).shape,
    )

    data_phi_arr = (
        None if data_phi is None else np.asarray(data_phi, dtype=float).ravel()
    )
    selected = resolve_phi_angles(args, config_manager, data_phi_angles=data_phi_arr)
    data["phi_angles_selected"] = selected

    # When the user explicitly passes --phi, actually subset the data so
    # the fit and plots see only those angles. (Config-based phi_filtering
    # is already applied by load_xpcs_data; this handles the CLI override
    # that the loader never saw.) Slice defensively: only when every
    # requested angle matches a data angle within tolerance — otherwise
    # keep all angles and warn, since fitting the wrong subset silently is
    # worse than fitting all.
    cli_phi = getattr(args, "phi", None)
    if cli_phi and data_phi_arr is not None:
        _subset_data_by_phi(data, data_phi_arr, [float(p) for p in cli_phi])

    logger.debug(
        "Phi selection resolved to %s angle(s): %s",
        0 if selected is None else len(selected),
        selected,
    )
    return data


def _subset_data_by_phi(
    data: dict[str, Any],
    data_phi: np.ndarray,
    requested: list[float],
    tol: float = 1.0,
) -> None:
    """Slice ``c2``/``phi`` arrays in-place to the requested angles.

    Matches each requested angle (mod-360 normalized) to a data-angle
    index within ``tol`` degrees. If any requested angle has no match, no
    slicing is performed (a warning is logged) — better to fit all angles
    than to silently fit the wrong subset.
    """
    norm_data = np.where((data_phi % 360) > 180, (data_phi % 360) - 360, data_phi % 360)
    indices: list[int] = []
    for ang in requested:
        norm_ang = ((ang % 360) + 360) % 360
        if norm_ang > 180:
            norm_ang -= 360
        diffs = np.abs(norm_data - norm_ang)
        j = int(np.argmin(diffs))
        if diffs[j] > tol:
            logger.warning(
                "--phi %.3f° has no matching data angle within %.1f° "
                "(closest %.3f°); not subsetting — fitting all angles.",
                ang,
                tol,
                float(data_phi[j]),
            )
            return
        indices.append(j)

    idx = np.asarray(sorted(set(indices)), dtype=int)
    c2 = _pick(data, _C2_KEYS)
    c2_arr = np.asarray(c2)
    # Phi axis is leading for stacked correlation data ([n_phi, t1, t2]).
    if c2_arr.ndim == 3 and c2_arr.shape[0] == len(data_phi):
        sliced = c2_arr[idx]
        for key in _C2_KEYS:
            if key in data:
                data[key] = sliced
        sliced_phi = data_phi[idx]
        for key in _PHI_KEYS:
            if key in data:
                data[key] = sliced_phi
        logger.info(
            "Subset data to %d CLI-requested phi angle(s): %s",
            len(idx),
            [float(a) for a in sliced_phi],
        )
    else:
        logger.warning(
            "c2 array shape %s does not have a leading phi axis matching "
            "%d data angles; --phi subsetting skipped.",
            c2_arr.shape,
            len(data_phi),
        )


def resolve_phi_angles(
    args: argparse.Namespace,
    config_manager: ConfigManager,
    data_phi_angles: np.ndarray | None = None,
) -> list[float] | None:
    """Determine phi angles from CLI args or configuration.

    Priority:
        1. ``args.phi``        -- explicit ``--phi`` list (real data).
        2. ``args.phi_angles`` -- comma-separated string (simulated mode).
        3. ``scattering.phi_angles`` in config.
        4. ``phi_filtering`` block against ``data_phi_angles``.
        5. ``data_phi_angles`` themselves (if present).
        6. ``None`` (no selection -- caller decides default).

    Args:
        args: Parsed CLI args; may have ``.phi``, ``.phi_angles``.
        config_manager: Configuration manager.
        data_phi_angles: Angles present in the loaded data (used for
            ``phi_filtering``). When omitted, filtering is skipped.

    Returns:
        Normalized phi angles in degrees, or ``None`` if nothing resolved.
    """
    phi_angles: list[float] | None = None

    # 1. Real-data CLI override (list[float]).
    cli_phi = getattr(args, "phi", None)
    if cli_phi:
        phi_angles = [float(a) for a in cli_phi]
        logger.debug("Phi angles from CLI --phi: %s", phi_angles)

    # 2. Simulated-data CLI override (comma-separated string).
    if phi_angles is None:
        cli_phi_str = getattr(args, "phi_angles", None)
        if isinstance(cli_phi_str, str) and cli_phi_str.strip():
            try:
                phi_angles = [
                    float(tok.strip())
                    for tok in cli_phi_str.split(",")
                    if tok.strip()
                ]
                logger.debug("Phi angles from CLI --phi-angles: %s", phi_angles)
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse --phi-angles={cli_phi_str!r}: {exc}"
                ) from exc

    cfg: dict[str, Any] = config_manager.get_config()

    # 3. scattering.phi_angles
    if phi_angles is None:
        scatter = cfg.get("scattering", {}) or {}
        scatter_phi = scatter.get("phi_angles")
        if scatter_phi:
            phi_angles = [float(a) for a in scatter_phi]
            logger.debug("Phi angles from config scattering.phi_angles: %s", phi_angles)

    # 4. phi_filtering against data.
    if phi_angles is None and data_phi_angles is not None:
        phi_cfg: dict[str, Any] = cfg.get("phi_filtering", {}) or {}
        if phi_cfg.get("enabled", False):
            filtered = _apply_phi_filtering(
                np.asarray(data_phi_angles, dtype=float), phi_cfg
            )
            if filtered is not None:
                phi_angles = filtered
                logger.debug("Phi angles from phi_filtering: %s", phi_angles)
            else:
                fallback = phi_cfg.get("fallback_to_all_angles", True)
                if fallback:
                    phi_angles = [float(a) for a in data_phi_angles]
                    logger.debug(
                        "phi_filtering matched nothing; falling back to all %d angles",
                        len(phi_angles),
                    )
                else:
                    raise ValueError(
                        "phi_filtering matched no data angles and "
                        "fallback_to_all_angles is false"
                    )

    # 5. Use raw data angles if still nothing.
    if phi_angles is None and data_phi_angles is not None:
        phi_angles = [float(a) for a in data_phi_angles]
        logger.debug(
            "No CLI/config phi source; using all %d data angles",
            len(phi_angles),
        )

    if phi_angles is None:
        logger.debug("No phi angle source resolved; returning None")
        return None

    phi_angles = [_norm_scalar(a) for a in phi_angles]
    logger.info("Analyzing phi angles: %s", phi_angles)
    return phi_angles


__all__ = ["load_and_validate_data", "resolve_phi_angles"]
