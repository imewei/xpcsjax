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
    mode: Literal["individual", "fourier", "constant", "auto"],
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
        The effective per-angle mode that produced the result. For ``'auto'``,
        read the dispatched mode from ``result.nlsq_diagnostics['per_angle_mode']``.
    layout : dict
        Layout descriptor with required keys:
          - ``n_physics`` : int
          - For fourier mode: ``fourier_order`` (K)

    Notes
    -----
    The Fourier coefficient convention used by both the packer
    (``_fit_joint_multi_phi``) and this evaluator is the **interleaved**
    layout produced by
    :class:`xpcsjax.optimization.nlsq.fourier_reparam.FourierReparameterizer`:

        ``[c_0, c_1, s_1, c_2, s_2, ..., c_K, s_K]``  (length ``2K + 1``)

    where ``c_0`` is the constant term and ``(c_k, s_k)`` are the
    (cosine, sine) amplitudes for harmonic ``k``. The evaluated series is

        ``f(phi) = c_0 + sum_{k=1..K} c_k cos(k phi) + s_k sin(k phi)``

    with ``phi`` in **degrees on input** (converted to radians internally).
    The single source of truth for this layout is
    ``FourierReparameterizer._compute_basis_matrix`` in
    ``xpcsjax/optimization/nlsq/fourier_reparam.py``.
    """
    phi = np.asarray(phi_angles, dtype=np.float64)
    n_phi = phi.size

    if mode == "constant":
        diag = result.nlsq_diagnostics or {}
        contrast = np.asarray(diag["contrast_per_angle_fixed"])
        offset = np.asarray(diag["offset_per_angle_fixed"])
        return {"contrast": contrast, "offset": offset}

    if mode == "individual":
        n_physics = int(layout["n_physics"])
        params = result.parameters
        contrast = params[n_physics : n_physics + n_phi]
        offset = params[n_physics + n_phi : n_physics + 2 * n_phi]
        return {"contrast": np.asarray(contrast), "offset": np.asarray(offset)}

    if mode == "fourier":
        n_physics = int(layout["n_physics"])
        K = int(layout["fourier_order"])
        basis_dim = 2 * K + 1  # constant + K cos + K sin
        params = result.parameters
        c_coeffs = params[n_physics : n_physics + basis_dim]
        o_coeffs = params[n_physics + basis_dim : n_physics + 2 * basis_dim]
        contrast = _evaluate_fourier_basis(c_coeffs, phi, K)
        offset = _evaluate_fourier_basis(o_coeffs, phi, K)
        return {"contrast": contrast, "offset": offset}

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


def _evaluate_fourier_basis(coeffs: np.ndarray, phi_deg: np.ndarray, K: int) -> np.ndarray:
    """Evaluate the truncated Fourier series at ``phi`` (degrees).

    Uses the canonical **interleaved** coefficient layout that matches
    :meth:`FourierReparameterizer._compute_basis_matrix`
    (see ``xpcsjax/optimization/nlsq/fourier_reparam.py``):

        ``[c_0, c_1, s_1, c_2, s_2, ..., c_K, s_K]``   (length ``2K + 1``)

    so that for ``k >= 1`` the cosine amplitude is at index ``2k - 1``
    and the sine amplitude is at index ``2k``. The evaluated series is

        ``f(phi) = c_0 + sum_{k=1..K} c_k cos(k phi) + s_k sin(k phi)``.

    Phi is provided in degrees and converted to radians internally to match
    the packer convention.
    """
    phi_rad = np.deg2rad(np.asarray(phi_deg, dtype=np.float64))
    coeffs = np.asarray(coeffs, dtype=np.float64)
    out = np.full_like(phi_rad, coeffs[0])
    for k in range(1, K + 1):
        out = out + coeffs[2 * k - 1] * np.cos(k * phi_rad)
        out = out + coeffs[2 * k] * np.sin(k * phi_rad)
    return out


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
