"""Headless data service — argparse-free core of the CLI data pipeline.

Worker-side module: imports JAX-bearing loaders (``xpcsjax.load_xpcs_data``)
and must only be imported by the worker, never the GUI process. Do NOT
re-export from ``xpcsjax.service.__init__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from xpcsjax import load_xpcs_data
from xpcsjax.data.angle_filtering import apply_angle_filtering_for_optimization
from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    from xpcsjax.config.manager import ConfigManager

logger = get_logger(__name__)

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


def _subset_data_by_phi(
    data: dict[str, Any],
    data_phi: np.ndarray,
    requested: list[float],
    tol: float = 1.0,
) -> None:
    """Slice ``c2``/``phi`` arrays in-place to the requested angles.

    Matches each requested angle (mod-360 normalized) to a data-angle index
    within ``tol`` degrees. If any requested angle has no match, no slicing is
    performed (a warning is logged) — better to fit all angles than to silently
    fit the wrong subset.
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
                "phi %.3f deg has no matching data angle within %.1f deg "
                "(closest %.3f deg); not subsetting — fitting all angles.",
                ang,
                tol,
                float(data_phi[j]),
            )
            return
        indices.append(j)

    idx = np.asarray(sorted(set(indices)), dtype=int)
    c2 = _pick(data, _C2_KEYS)
    c2_arr = np.asarray(c2)
    if c2_arr.ndim == 3 and c2_arr.shape[0] == len(data_phi):
        sliced = c2_arr[idx]
        for key in _C2_KEYS:
            if key in data:
                data[key] = sliced
        sliced_phi = data_phi[idx]
        for key in _PHI_KEYS:
            if key in data:
                data[key] = sliced_phi
        logger.info("Subset data to %d requested phi angle(s): %s", len(idx), [float(a) for a in sliced_phi])
    else:
        logger.warning(
            "c2 array shape %s has no leading phi axis matching %d data angles; "
            "phi subsetting skipped.",
            c2_arr.shape,
            len(data_phi),
        )


def load_dataset(
    config_manager: ConfigManager,
    *,
    phi_subset: list[float] | None = None,
) -> dict[str, Any]:
    """Load XPCS data and apply phi filtering.

    Argparse-free core of ``cli.data_pipeline.load_and_validate_data``. The
    ``phi_subset`` parameter replaces the legacy ``args.phi`` override.

    Parameters
    ----------
    config_manager : ConfigManager
        Configuration manager with ``get_config()`` method.
    phi_subset : list of float, optional
        Phi angles to subset the data to. If provided and data contains a
        phi-axis, data is sliced in-place. ``None`` keeps all angles.

    Returns
    -------
    dict
        XPCS data dict with keys including ``c2_exp``/``c2``,
        ``phi_angles_list``/``phi_angles``, ``t1``, ``t2``.

    Raises
    ------
    ValueError
        If no correlation matrix is found in the loader output.
    """
    cfg = config_manager.get_config()
    analysis_mode = cfg.get("analysis_mode", "<unknown>")
    data_type = cfg.get("data_type")
    if data_type is None:
        data_type = (cfg.get("experimental_data") or {}).get("data_type")
    if data_type not in (None, "aps_old", "aps_u"):
        logger.warning("Unrecognized data_type=%r (expected 'aps_old' or 'aps_u')", data_type)

    logger.info("Loading XPCS data (analysis_mode=%s, data_type=%s)", analysis_mode, data_type)

    data: dict[str, Any] = load_xpcs_data(config_dict=cfg)

    c2 = _pick(data, _C2_KEYS)
    if c2 is None:
        raise ValueError(
            f"load_xpcs_data returned no correlation matrix; expected one of {_C2_KEYS}"
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
    data_phi_arr = None if data_phi is None else np.asarray(data_phi, dtype=float).ravel()

    if phi_subset and data_phi_arr is not None:
        _subset_data_by_phi(data, data_phi_arr, [float(p) for p in phi_subset])

    data = apply_angle_filtering_for_optimization(data, config_manager)
    return data


def resolve_phi_angles(
    config_manager: ConfigManager,
    *,
    cli_phi: list[float] | None = None,
    phi_angles_str: str | None = None,
) -> list[float] | None:
    """Determine phi angles from explicit inputs or configuration.

    Argparse-free core of ``cli.data_pipeline.resolve_phi_angles``. Priority:
    ``cli_phi`` (real-data list) > ``phi_angles_str`` (simulated, comma string)
    > ``scattering.phi_angles`` in config > ``None``. Returns angles normalized
    to ``[-180, 180]``.

    Parameters
    ----------
    config_manager : ConfigManager
        Configuration manager with ``get_config()`` method.
    cli_phi : list of float, optional
        Explicit phi angles from CLI (``--phi``). Takes precedence over all
        other sources.
    phi_angles_str : str, optional
        Comma-separated phi angle string (simulated-data ``--phi-angles``).
        Takes precedence over config source.

    Returns
    -------
    list of float or None
        Phi angles normalized to ``[-180, 180]``, or ``None`` if no source
        resolves.

    Raises
    ------
    ValueError
        If ``phi_angles_str`` cannot be parsed as comma-separated floats.
    """
    phi_angles: list[float] | None = None

    if cli_phi:
        phi_angles = [float(a) for a in cli_phi]
        logger.debug("Phi angles from explicit cli_phi: %s", phi_angles)

    if phi_angles is None and isinstance(phi_angles_str, str) and phi_angles_str.strip():
        try:
            phi_angles = [float(tok.strip()) for tok in phi_angles_str.split(",") if tok.strip()]
            logger.debug("Phi angles from phi_angles_str: %s", phi_angles)
        except ValueError as exc:
            raise ValueError(f"Could not parse phi_angles_str={phi_angles_str!r}: {exc}") from exc

    if phi_angles is None:
        scatter = config_manager.get_config().get("scattering", {}) or {}
        scatter_phi = scatter.get("phi_angles")
        if scatter_phi:
            phi_angles = [float(a) for a in scatter_phi]
            logger.debug("Phi angles from config scattering.phi_angles: %s", phi_angles)

    if phi_angles is None:
        logger.debug("No phi angle source resolved; returning None")
        return None

    normalized = [_norm_scalar(a) for a in phi_angles]
    logger.info("Analyzing phi angles: %s", normalized)
    return normalized
