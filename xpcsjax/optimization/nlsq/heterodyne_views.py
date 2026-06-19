"""Post-hoc views of heterodyne joint-fit results.

These are pure functions of (OptimizationResult, layout, phi_angles).
They reconstruct per-angle quantities that aren't stored in the result.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from xpcsjax.optimization.nlsq.results import OptimizationResult


def reconstruct_per_angle_scaling(
    result: OptimizationResult,
    phi_angles: np.ndarray,
    mode: Literal["individual", "constant", "averaged", "auto"],
    layout: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Return ``{'contrast': (n_phi,), 'offset': (n_phi,)}`` from fit parameters.

    Pure function of the result + layout descriptor. No I/O.

    Parameters
    ----------
    result : OptimizationResult
        The fit result whose ``parameters`` vector encodes the scaling.
    phi_angles : np.ndarray
        Phi angles in degrees, shape ``(n_phi,)``.
    mode : str
        The effective per-angle mode that produced the result —
        ``'individual'``, ``'constant'``, ``'averaged'``, or ``'auto'``. For
        ``'auto'``, read the dispatched mode from
        ``result.nlsq_diagnostics['per_angle_mode']`` (which itself resolves to
        one of the three concrete variants).
    layout : dict
        Layout descriptor with required keys:
          - ``n_physics`` : int
    """
    phi = np.asarray(phi_angles, dtype=np.float64)
    n_phi = phi.size

    if mode == "constant":
        diag = result.nlsq_diagnostics or {}
        contrast = np.asarray(diag["contrast_per_angle_fixed"])
        offset = np.asarray(diag["offset_per_angle_fixed"])
        return {"contrast": contrast, "offset": offset}

    if mode == "individual":
        # Canonical scaling-first layout: [c_0..n_phi-1, o_0..n_phi-1, physics...]
        # — the scaling HEAD precedes the physics TAIL (Tasks 3-12 of
        # per-angle-mode unification).  n_physics is only used to verify the
        # layout via layout["n_physics"] but the slice reads from the HEAD.
        params = result.parameters
        contrast = params[:n_phi]
        offset = params[n_phi : 2 * n_phi]
        return {"contrast": np.asarray(contrast), "offset": np.asarray(offset)}

    if mode == "averaged":
        # Canonical scaling-first averaged layout: [c_avg, o_avg, physics...].
        # One fitted (contrast, offset) pair shared across all angles. Prefer
        # the fitted scalars stored in diagnostics; fall back to the scaling
        # HEAD params[0]/[1] (NOT the tail — physics occupies the tail). Broadcast
        # to n_phi so the return contract matches individual/constant mode.
        diag = result.nlsq_diagnostics or {}
        params = np.asarray(result.parameters, dtype=np.float64)
        # Scaling scalars come from diagnostics regardless of layout. The
        # positional fallback depends on the layout: scaling-first puts (c, o) in
        # the HEAD, the legacy physics-first averaged path puts them in the TAIL
        # (audit #1). Default scaling-first when the marker is absent.
        scaling_first = bool(diag.get("scaling_first", True))
        fallback_c, fallback_o = (
            (params[0], params[1]) if scaling_first else (params[-2], params[-1])
        )
        c = float(diag.get("averaged_contrast", fallback_c))
        o = float(diag.get("averaged_offset", fallback_o))
        return {
            "contrast": np.full(n_phi, c, dtype=np.float64),
            "offset": np.full(n_phi, o, dtype=np.float64),
        }

    if mode == "auto":
        diag = result.nlsq_diagnostics or {}
        actual_mode = diag.get("per_angle_mode")
        if actual_mode is None or actual_mode == "auto":
            raise ValueError(
                "Cannot reconstruct from 'auto' mode without knowing the "
                "dispatched effective mode; nlsq_diagnostics['per_angle_mode'] "
                "is missing or unresolved."
            )
        return reconstruct_per_angle_scaling(result, phi, actual_mode, layout)

    raise ValueError(f"unknown mode: {mode!r}")


def per_angle_chi2(result: OptimizationResult) -> np.ndarray:
    """Return per-angle chi^2 from ``nlsq_diagnostics``.

    Raises
    ------
    ValueError
        If ``chi2_per_angle`` is not populated (e.g. this is not a heterodyne fit).
    """
    diag = result.nlsq_diagnostics or {}
    if "chi2_per_angle" not in diag:
        raise ValueError("chi2_per_angle not in nlsq_diagnostics — was this a heterodyne fit?")
    return np.asarray(diag["chi2_per_angle"])
