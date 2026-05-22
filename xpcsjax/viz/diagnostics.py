"""Diagonal-overlay diagnostic helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DiagonalOverlayResult", "compute_diagonal_overlay_stats"]


@dataclass
class DiagonalOverlayResult:
    """Per-angle diagonal trace statistics.

    Attributes
    ----------
    phi_index
        Angle index that was sliced.
    raw_diagonal
        ``np.diag(c2_exp[phi_index])`` — experimental c2 along t1=t2.
    fitted_diagonal
        ``np.diag(c2_fit[phi_index])`` — fitted c2 along t1=t2.
    raw_variance
        Sample variance of ``raw_diagonal`` (NaN-safe).
    fitted_variance
        Sample variance of ``fitted_diagonal`` (NaN-safe).
    fitted_rmse
        ``sqrt(nanmean((fitted_diagonal - raw_diagonal) ** 2))``.
    """

    phi_index: int
    raw_diagonal: np.ndarray
    fitted_diagonal: np.ndarray
    raw_variance: float
    fitted_variance: float
    fitted_rmse: float


def compute_diagonal_overlay_stats(
    c2_exp: np.ndarray,
    c2_fit: np.ndarray,
    *,
    phi_index: int = 0,
) -> DiagonalOverlayResult:
    """Extract diagonals of c2_exp and c2_fit at one angle.

    Parameters
    ----------
    c2_exp
        Experimental c2 surface, shape ``(n_phi, n_t1, n_t2)``.
    c2_fit
        Fitted c2 surface, same shape as ``c2_exp``.
    phi_index
        Which angle to slice. Raises ``IndexError`` if out of bounds.
    """
    if c2_exp.ndim != 3:
        raise ValueError(f"c2_exp must be 3-D (n_phi, n_t1, n_t2); got shape {c2_exp.shape}")
    if c2_fit.shape != c2_exp.shape:
        raise ValueError(
            f"c2_fit.shape {c2_fit.shape} must equal c2_exp.shape {c2_exp.shape}"
        )
    if phi_index < 0 or phi_index >= c2_exp.shape[0]:
        raise IndexError(
            f"phi_index={phi_index} out of bounds for c2_exp with {c2_exp.shape[0]} angles"
        )
    raw_diag = np.diag(c2_exp[phi_index])
    fitted_diag = np.diag(c2_fit[phi_index])
    # ddof=1 matches the docstring claim of "sample variance" (N-1 denominator).
    # Guard against n<2 to avoid division-by-zero; return NaN for degenerate inputs.
    n_raw = int(np.isfinite(raw_diag).sum())
    n_fit = int(np.isfinite(fitted_diag).sum())
    return DiagonalOverlayResult(
        phi_index=phi_index,
        raw_diagonal=raw_diag,
        fitted_diagonal=fitted_diag,
        raw_variance=float(np.nanvar(raw_diag, ddof=1)) if n_raw > 1 else float("nan"),
        fitted_variance=float(np.nanvar(fitted_diag, ddof=1)) if n_fit > 1 else float("nan"),
        fitted_rmse=float(np.sqrt(np.nanmean((fitted_diag - raw_diag) ** 2))),
    )
