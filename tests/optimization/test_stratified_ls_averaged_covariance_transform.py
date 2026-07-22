"""Regression: the averaged-mode inverse covariance transform in
``fit_with_stratified_least_squares`` must use the broadcast (all-ones) Jacobian,
not an identity-scaled diagonal that drops cross-angle correlation and the
contrast-offset cross block entirely.

Root cause: every per-angle contrast (and per-angle offset) is a literal
broadcast copy of the single fitted scalar, so the correct Jacobian is
``J[:n_phi, 0] = 1`` / ``J[n_phi:2*n_phi, 1] = 1`` / physical block identity,
giving ``pcov_expanded = J @ pcov @ J.T``. The prior manual block assignment
(``np.eye(n_phi) * pcov[0, 0]``) is only correct on the diagonal and silently
zeroes the contrast-contrast cross-angle terms and the contrast-offset cross
block.

This test pins the math with a hand-computed n_phi=2, n_physical=1 example
(the exact case suggested in the audit finding) and confirms the source no
longer contains the disproven identity-diagonal shortcut.
"""

from __future__ import annotations

import inspect

import numpy as np

from xpcsjax.optimization.nlsq.strategies import stratified_ls


def _broadcast_jacobian_transform(pcov: np.ndarray, n_phi: int, n_physical: int) -> np.ndarray:
    """Exact transform now used in ``fit_with_stratified_least_squares``'s
    averaged-mode inverse-covariance block (mirrors hybrid_streaming.py)."""
    n_constant_total = 2 + n_physical
    n_total = 2 * n_phi + n_physical
    J_full = np.zeros((n_total, n_constant_total))
    J_full[:n_phi, 0] = 1.0
    J_full[n_phi : 2 * n_phi, 1] = 1.0
    J_full[2 * n_phi :, 2:] = np.eye(n_physical)
    return J_full @ pcov @ J_full.T


def test_hand_computed_n_phi2_n_physical1():
    """n_phi=2, n_physical=1: pcov is 3x3 over [contrast, offset, D0]."""
    pcov = np.array(
        [
            [2.0, 0.5, 0.1],  # var(contrast), cov(contrast, offset), cov(contrast, D0)
            [0.5, 3.0, 0.2],  # cov(offset, contrast), var(offset), cov(offset, D0)
            [0.1, 0.2, 4.0],  # cov(D0, contrast), cov(D0, offset), var(D0)
        ]
    )
    n_phi, n_physical = 2, 1
    pcov_expanded = _broadcast_jacobian_transform(pcov, n_phi, n_physical)
    assert pcov_expanded.shape == (2 * n_phi + n_physical, 2 * n_phi + n_physical)

    # Contrast-contrast block: every pair of per-angle contrasts (including a
    # value with itself) is the SAME broadcast scalar -> covariance = pcov[0, 0]
    # everywhere, NOT an identity-scaled diagonal (which would zero the
    # off-diagonal cross-angle terms).
    contrast_block = pcov_expanded[:n_phi, :n_phi]
    assert np.allclose(contrast_block, pcov[0, 0])
    assert not np.allclose(contrast_block, np.eye(n_phi) * pcov[0, 0]), (
        "contrast-contrast block regressed to the buggy identity-scaled diagonal"
    )

    # Offset-offset block: same broadcast argument -> pcov[1, 1] everywhere.
    offset_block = pcov_expanded[n_phi : 2 * n_phi, n_phi : 2 * n_phi]
    assert np.allclose(offset_block, pcov[1, 1])

    # Contrast-offset cross block: previously dropped to zero entirely; must
    # now be pcov[0, 1] (the fitted contrast-offset covariance) everywhere.
    cross_block = pcov_expanded[:n_phi, n_phi : 2 * n_phi]
    assert np.allclose(cross_block, pcov[0, 1])
    assert not np.allclose(cross_block, 0.0), (
        "contrast-offset cross block regressed to the buggy all-zero fallback"
    )

    # Physical block passes through unchanged (identity sub-block of J).
    physical_block = pcov_expanded[2 * n_phi :, 2 * n_phi :]
    assert np.allclose(physical_block, pcov[2:, 2:])


def test_source_no_longer_uses_identity_diagonal_shortcut():
    """Wiring: the averaged-mode inverse block must use the J_full broadcast
    transform, not the disproven ``np.eye(n_phi) * pcov[0, 0]`` shortcut."""
    src = inspect.getsource(stratified_ls.fit_with_stratified_least_squares)
    assert "J_full" in src, (
        "averaged-mode covariance expansion no longer uses the broadcast Jacobian transform"
    )
    assert "np.eye(n_phi) * pcov[0, 0]" not in src, (
        "averaged-mode covariance expansion regressed to the identity-scaled diagonal shortcut"
    )
