"""Diagonal-overlay diagnostic helpers.

The diagonal of a two-time correlation surface ``c2[phi]`` (the ``t1 == t2``
trace) is the most sensitive single slice for spotting amplitude or contrast
mismatch between an experimental surface and its NLSQ fit. This module extracts
that trace for one angle and reports NaN-safe variance and RMSE statistics so a
caller can quantify per-angle fit quality without rendering a figure.

See Also
--------
xpcsjax.viz.nlsq_plots.plot_residual_map : Plots the same diagonal trace.
xpcsjax.fit_nlsq : Produces the fitted surfaces compared here.
"""

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
    """Extract the t₁=t₂ diagonals of two c2 surfaces at one angle.

    Slices both surfaces at ``phi_index``, takes the main diagonal of each, and
    computes NaN-safe sample variance (``ddof=1``) and the diagonal RMSE between
    them.

    Parameters
    ----------
    c2_exp
        Experimental c2 surface, shape ``(n_phi, n_t1, n_t2)``.
    c2_fit
        Fitted c2 surface, same shape as ``c2_exp``.
    phi_index
        Which angle to slice (keyword-only).

    Returns
    -------
    DiagonalOverlayResult
        The sliced diagonals plus their per-trace variances and the diagonal
        RMSE. See :class:`DiagonalOverlayResult` for field semantics.

    Raises
    ------
    ValueError
        If ``c2_exp`` is not 3-D, or if ``c2_fit.shape`` differs from
        ``c2_exp.shape``.
    IndexError
        If ``phi_index`` is negative or ``>= n_phi``.

    Notes
    -----
    Variance is the sample variance (``ddof=1``, N-1 denominator) over finite
    entries; a trace with fewer than two finite values yields ``nan``. The RMSE
    is computed over diagonal positions where *both* traces are finite, so a
    single ``inf`` cannot poison the result; when no such position exists the
    RMSE is ``nan``.

    See Also
    --------
    DiagonalOverlayResult : The returned dataclass.
    xpcsjax.viz.nlsq_plots.plot_residual_map : Visual counterpart of this trace.

    Examples
    --------
    >>> import numpy as np
    >>> from xpcsjax.viz import compute_diagonal_overlay_stats
    >>> c2_exp = np.full((2, 32, 32), 1.2)
    >>> c2_fit = c2_exp + 1e-3
    >>> stats = compute_diagonal_overlay_stats(c2_exp, c2_fit, phi_index=0)
    >>> round(stats.fitted_rmse, 4)
    0.001
    """
    if c2_exp.ndim != 3:
        raise ValueError(f"c2_exp must be 3-D (n_phi, n_t1, n_t2); got shape {c2_exp.shape}")
    if c2_fit.shape != c2_exp.shape:
        raise ValueError(f"c2_fit.shape {c2_fit.shape} must equal c2_exp.shape {c2_exp.shape}")
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
    # RMSE over pairs where BOTH diagonals are finite — nanmean ignores NaN but
    # not inf, so an inf entry would otherwise poison the mean.
    pair_mask = np.isfinite(raw_diag) & np.isfinite(fitted_diag)
    if pair_mask.any():
        diff = fitted_diag[pair_mask] - raw_diag[pair_mask]
        fitted_rmse = float(np.sqrt(np.mean(diff**2)))
    else:
        fitted_rmse = float("nan")
    return DiagonalOverlayResult(
        phi_index=phi_index,
        raw_diagonal=raw_diag,
        fitted_diagonal=fitted_diag,
        raw_variance=float(np.nanvar(raw_diag, ddof=1)) if n_raw > 1 else float("nan"),
        fitted_variance=float(np.nanvar(fitted_diag, ddof=1)) if n_fit > 1 else float("nan"),
        fitted_rmse=fitted_rmse,
    )
