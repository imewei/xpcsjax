"""Regression tests for ``XPCSDataFilter._calculate_matrix_quality_score``.

The quality score *gates* angle selection (``quality_score < quality_threshold``
drops the angle from the fit in ``_apply_quality_filtering``), so a spurious
penalty here is silent data loss, not a cosmetic reporting nit.

Bug: the diagonal check read ``diagonal[0]`` — the EXACT tau=0 main-diagonal
element. For raw two-time XPCS that element is the self-correlation / shot-noise
spike (routinely ~2.4, far above the Siegert ceiling g2(0) <= 2.0) and is
excluded from analysis. Evaluating it penalized every clean angle of valid data.
The Siegert relation holds at the smallest *non-zero* lag (first off-diagonal),
mirroring the already-correct check in ``data/validation.py``.
"""

import numpy as np
import pytest

from xpcsjax.data.filtering_utils import XPCSDataFilter


def _clean_matrix_with_raw_spike(n: int = 24, off_diag: float = 1.3, spike: float = 2.4):
    """A clean, symmetric correlation matrix carrying the raw tau=0 spike.

    Off-diagonal (real lags) sit at a physical g2 ~ ``off_diag`` (within the
    Siegert band); the main diagonal carries the excluded self-correlation
    ``spike`` (> 2.0). This is what a raw HDF5-reconstructed matrix looks like
    before diagonal correction.
    """
    matrix = np.full((n, n), off_diag, dtype=np.float64)
    np.fill_diagonal(matrix, spike)
    return matrix


def test_clean_matrix_with_raw_tau0_spike_is_not_penalized():
    """A valid matrix must score full marks despite the raw tau=0 spike."""
    f = XPCSDataFilter()
    matrix = _clean_matrix_with_raw_spike()

    score = f._calculate_matrix_quality_score(matrix)

    # finite(0.4) + diagonal(0.3) + symmetry(0.2) + range(0.1) = 1.0 when the
    # diagonal check looks at the real first-lag value (~1.3) rather than the
    # spike. The buggy version reads diagonal[0]=2.4 (> 2.0), drops diagonal
    # quality to 0.5*0.3=0.15, and yields ~0.85 — below a strict threshold.
    assert score >= 0.99, f"clean matrix wrongly penalized: score={score:.3f}"


def test_genuinely_overnormalized_first_lag_is_still_penalized():
    """The fix must not mask real problems: lag-1 g2 > 2.0 stays penalized."""
    f = XPCSDataFilter()
    # First off-diagonal at 2.6 (> Siegert ceiling) => over-normalized data.
    matrix = _clean_matrix_with_raw_spike(off_diag=2.6, spike=2.6)

    score = f._calculate_matrix_quality_score(matrix)

    # diagonal quality must drop (0.5), so total cannot reach the full 1.0.
    assert score < 0.95, f"over-normalized matrix should be penalized: score={score:.3f}"


def test_single_element_matrix_does_not_crash():
    """1x1 matrices have no off-diagonal lag; must fall back gracefully."""
    f = XPCSDataFilter()
    matrix = np.array([[1.5]], dtype=np.float64)

    score = f._calculate_matrix_quality_score(matrix)

    assert 0.0 <= score <= 1.0


def test_single_element_matrix_uses_neutral_diagonal_quality():
    """A <=1-frame matrix has no lag-1 to check against the Siegert ceiling;
    the fix must score it neutrally regardless of the diagonal value, not by
    evaluating diagonal[0] (which is the excluded tau=0 spike, not lag-1 data).

    Discriminating value: 1.3 sits *inside* the pre-fix code's [0.5, 2.0]
    Siegert band, so the bug (reading diagonal[0]=1.3 through that check)
    would score diagonal_quality=1.0 -- coincidentally "clean" for a reading
    that isn't lag-1 data at all -- giving a perfect 1.0 total. The fix's
    constant neutral 0.5 gives 0.85 instead, regardless of the actual value.
    """
    f = XPCSDataFilter()
    matrix = np.array([[1.3]], dtype=np.float64)

    score = f._calculate_matrix_quality_score(matrix)

    # finite(0.4) + diagonal(neutral 0.5*0.3=0.15) + symmetry(0.2, 1x1 is
    # trivially symmetric) + range(0.1, mean=1.3 is within [0.1,5.0]) = 0.85.
    assert score == pytest.approx(0.85), f"unexpected score for degenerate matrix: {score:.3f}"
