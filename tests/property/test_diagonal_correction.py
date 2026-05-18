"""Diagonal correction is mandatory for both physics models.

Property: after correction, c2[i, i] is replaced with values interpolated from
adjacent off-diagonal entries — regardless of method — so the autocorrelation
peak is removed.
"""

import numpy as np
import pytest

from xpcsjax.core.diagonal_correction import apply_diagonal_correction


@pytest.mark.parametrize("method", ["basic", "statistical", "interpolation"])
def test_diagonal_is_replaced(method):
    rng = np.random.default_rng(seed=42)
    N = 32
    c2 = rng.uniform(0.5, 1.5, size=(N, N))
    c2 = (c2 + c2.T) / 2  # symmetrize
    # Spike the diagonal to simulate the autocorrelation peak
    c2[np.arange(N), np.arange(N)] = 5.0

    corrected = apply_diagonal_correction(c2, method=method)

    diag_max = np.max(np.abs(np.diag(corrected)))
    assert diag_max < 2.5, (
        f"method={method}: diagonal still contains autocorr-peak-magnitude values "
        f"(max abs diag = {diag_max})"
    )


def test_off_diagonal_preserved():
    """Correction must NOT modify off-diagonal entries."""
    rng = np.random.default_rng(seed=7)
    N = 16
    c2 = rng.uniform(0.5, 1.5, size=(N, N))
    c2 = (c2 + c2.T) / 2

    corrected = apply_diagonal_correction(c2.copy(), method="basic")

    off_diag_mask = ~np.eye(N, dtype=bool)
    np.testing.assert_allclose(corrected[off_diag_mask], c2[off_diag_mask])
