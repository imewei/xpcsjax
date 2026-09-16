"""Result-assembly helpers shared by the homodyne fit tiers.

Extracted from ``wrapper.py`` (codebase review 2026-09-15, three-brain
follow-up) so the route modules (``wrapper_out_of_core_route``,
``wrapper_stratified_route``) no longer have to import them back out of the
module that imports the routes. Pure functions over solver output; no
wrapper state.
"""

from __future__ import annotations

import numpy as np

from xpcsjax.optimization.nlsq.anti_degeneracy_diagnostics import (
    assemble_anti_degeneracy_diagnostics,
)

# L5 (shear weighting) is laminar_flow-only; every laminar path that does not run
# it reports this sentinel (the heterodyne surfaces translate it -- see CLAUDE.md).
_LAMINAR_L5_INACTIVE = "laminar_flow_inactive"

__all__ = [
    "_LAMINAR_L5_INACTIVE",
    "_info_cov_placeholder",
    "_laminar_anti_degeneracy_block",
    "_uncertainties_from_pcov",
]


def _laminar_anti_degeneracy_block(anti_degeneracy_info: dict | None) -> dict:
    """Build the symmetric anti-degeneracy diagnostics block for a laminar result.

    Reads from the solver's ``info['anti_degeneracy']`` dict (or ``None``).
    Presence-based and honest: a layer is reported active only when its optimizer
    actually ran and set its sub-key in ``info['anti_degeneracy']``; otherwise the
    laminar inactive marker. This single rule is correct for every non-in-memory
    laminar return path:

    - HYBRID_STREAMING threads honest ``"hierarchical"``/``"regularization"``/
      ``"shear_weighting"``/``"gradient_monitor"`` sub-keys only when the
      corresponding optimizer ran -> reported active.
    - stratified-LS carries only ``mode``/``controller_diagnostics`` (no layer
      sub-keys) -> honestly inactive.
    - sequential / out-of-core run no anti-degeneracy and pass ``None`` -> markers.

    Diagnostics-only: never reads or writes popt/pcov/chi2.
    """
    ad = anti_degeneracy_info or {}
    return assemble_anti_degeneracy_diagnostics(
        hierarchical_active="hierarchical" in ad,
        regularization_active="regularization" in ad,
        shear_weighting=ad.get("shear_weighting", _LAMINAR_L5_INACTIVE),
        gradient_monitor=ad.get("gradient_monitor"),
    )


def _uncertainties_from_pcov(
    pcov: np.ndarray | None, n_params: int, *, is_placeholder: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """``(covariance, uncertainties)`` under the ONE rule every path shares.

    ``is_placeholder`` (the strategy's ``info["covariance_is_placeholder"]``,
    set on the reduced solver covariance before any fixed-slot / scaling
    expansion) or a shape mismatch -> all-NaN covariance and uncertainties.
    Otherwise ``sqrt(diag)``: structural exact-zero rows (fixed physical
    slots, frozen constant-mode scaling) read as 0.0, a non-finite or negative
    variance reads as NaN. No floor — the old ``1e-5`` regularisation floor
    (``recovery.safe_uncertainties_from_pcov``) fabricated a tiny "known"
    uncertainty for singular / null-space directions.
    """
    nan_cov = np.full((n_params, n_params), np.nan, dtype=np.float64)
    nan_unc = np.full(n_params, np.nan, dtype=np.float64)
    if is_placeholder or pcov is None:
        return nan_cov, nan_unc
    pcov = np.asarray(pcov, dtype=np.float64)
    if pcov.shape != (n_params, n_params):
        return nan_cov, nan_unc
    diag = np.diag(pcov)
    unc = np.where(np.isfinite(diag) & (diag >= 0.0), np.sqrt(np.clip(diag, 0.0, None)), np.nan)
    return pcov, np.asarray(unc, dtype=np.float64)


def _info_cov_placeholder(info: dict | None) -> bool:
    """Read ``covariance_is_placeholder`` from a strategy ``info`` (top level or nested)."""
    if not info:
        return False
    nested = info.get("anti_degeneracy") or {}
    return bool(
        info.get("covariance_is_placeholder", False)
        or nested.get("covariance_is_placeholder", False)
    )
