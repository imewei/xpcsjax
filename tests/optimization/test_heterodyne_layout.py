"""Unit tests for physics-first ⇄ scaling-first layout conversion (Phase 2.2).

Pure-helper tests only — NO engine wiring, NO full fit. Covers all three
in-scope modes (``fixed_constant``, ``auto_averaged``, ``individual``):

* round-trip identity ``scaling_first_to_physics_first(physics_first_to_scaling_first(v)) == v``
* hand-checked element placement per mode
* covariance permuter: own-inverse under inverse permutation, symmetry-preserving
* ``fixed_constant`` is identity on the physics vector
* ``auto_averaged`` is a broadcast/compress (not a permutation) — permutation raises
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.heterodyne_layout import (
    IN_SCOPE_MODES,
    permute_cov,
    physics_first_to_scaling_first,
    scaling_first_permutation,
    scaling_first_to_physics_first,
)

# Representative two_component-like sizing (14-param model in the wild; the
# *varying* physics count is what enters the optimizer vector — use 14 here to
# exercise a realistic n_physics).
N_PHYSICS = 14
N_PHI = 3


def _physics_first_vec(mode: str, n_physics: int, n_phi: int) -> np.ndarray:
    """Build a distinct-valued physics-first vector for *mode*."""
    physics = 100.0 + np.arange(n_physics, dtype=np.float64)  # 100..100+n_physics-1
    if mode == "fixed_constant":
        return physics
    if mode == "auto_averaged":
        return np.concatenate([physics, [0.30, 1.05]])  # contrast_scalar, offset_scalar
    # individual: contrast(n_phi) then offset(n_phi)
    contrast = 0.30 + 0.01 * np.arange(n_phi, dtype=np.float64)
    offset = 1.00 + 0.01 * np.arange(n_phi, dtype=np.float64)
    return np.concatenate([physics, contrast, offset])


# ---------------------------------------------------------------------------
# Round-trip identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", IN_SCOPE_MODES)
def test_roundtrip_identity(mode: str) -> None:
    v = _physics_first_vec(mode, N_PHYSICS, N_PHI)
    sf = physics_first_to_scaling_first(v, n_physics=N_PHYSICS, mode=mode, n_phi=N_PHI)
    back = scaling_first_to_physics_first(sf, n_physics=N_PHYSICS, mode=mode, n_phi=N_PHI)
    np.testing.assert_array_equal(back, v)


@pytest.mark.parametrize("mode", IN_SCOPE_MODES)
@pytest.mark.parametrize("n_phi", [1, 2, 3, 5])
def test_roundtrip_identity_varied_nphi(mode: str, n_phi: int) -> None:
    v = _physics_first_vec(mode, N_PHYSICS, n_phi)
    sf = physics_first_to_scaling_first(v, n_physics=N_PHYSICS, mode=mode, n_phi=n_phi)
    back = scaling_first_to_physics_first(sf, n_physics=N_PHYSICS, mode=mode, n_phi=n_phi)
    np.testing.assert_array_equal(back, v)


# ---------------------------------------------------------------------------
# Output length / shape
# ---------------------------------------------------------------------------
def test_scaling_first_lengths() -> None:
    # fixed_constant: physics only on both sides
    v = _physics_first_vec("fixed_constant", N_PHYSICS, N_PHI)
    sf = physics_first_to_scaling_first(v, n_physics=N_PHYSICS, mode="fixed_constant", n_phi=N_PHI)
    assert sf.shape == (N_PHYSICS,)
    # auto_averaged: physics-first len = n_physics + 2; scaling-first = 2*n_phi + n_physics
    v = _physics_first_vec("auto_averaged", N_PHYSICS, N_PHI)
    assert v.shape == (N_PHYSICS + 2,)
    sf = physics_first_to_scaling_first(v, n_physics=N_PHYSICS, mode="auto_averaged", n_phi=N_PHI)
    assert sf.shape == (2 * N_PHI + N_PHYSICS,)
    # individual: physics-first = n_physics + 2*n_phi; scaling-first = 2*n_phi + n_physics (same)
    v = _physics_first_vec("individual", N_PHYSICS, N_PHI)
    assert v.shape == (N_PHYSICS + 2 * N_PHI,)
    sf = physics_first_to_scaling_first(v, n_physics=N_PHYSICS, mode="individual", n_phi=N_PHI)
    assert sf.shape == (2 * N_PHI + N_PHYSICS,)


# ---------------------------------------------------------------------------
# Hand-checked element placement (the load-bearing correctness checks)
# ---------------------------------------------------------------------------
def test_individual_hand_checked_placement() -> None:
    # n_physics=14, n_phi=3, individual.
    physics = 100.0 + np.arange(14.0)
    contrast = np.array([0.30, 0.31, 0.32])
    offset = np.array([1.00, 1.01, 1.02])
    v = np.concatenate([physics, contrast, offset])  # physics-first
    sf = physics_first_to_scaling_first(v, n_physics=14, mode="individual", n_phi=3)
    # scaling-first = [contrast(3) | offset(3) | physics(14)]
    np.testing.assert_array_equal(sf[:3], contrast)
    np.testing.assert_array_equal(sf[3:6], offset)
    np.testing.assert_array_equal(sf[6:], physics)


def test_auto_averaged_hand_checked_broadcast() -> None:
    # n_physics=14, n_phi=3, auto_averaged. 2 scalars must broadcast to 3 each.
    physics = 100.0 + np.arange(14.0)
    v = np.concatenate([physics, [0.42, 1.07]])  # [physics | contrast_scalar | offset_scalar]
    sf = physics_first_to_scaling_first(v, n_physics=14, mode="auto_averaged", n_phi=3)
    np.testing.assert_array_equal(sf[:3], np.full(3, 0.42))  # contrast broadcast
    np.testing.assert_array_equal(sf[3:6], np.full(3, 1.07))  # offset broadcast
    np.testing.assert_array_equal(sf[6:], physics)
    # compress back to 2 scalars
    back = scaling_first_to_physics_first(sf, n_physics=14, mode="auto_averaged", n_phi=3)
    np.testing.assert_array_equal(back, v)


def test_fixed_constant_is_identity() -> None:
    physics = 100.0 + np.arange(14.0)
    sf = physics_first_to_scaling_first(physics, n_physics=14, mode="fixed_constant", n_phi=3)
    np.testing.assert_array_equal(sf, physics)
    assert sf is not physics  # returns a copy, not the same object
    back = scaling_first_to_physics_first(sf, n_physics=14, mode="fixed_constant", n_phi=3)
    np.testing.assert_array_equal(back, physics)


# ---------------------------------------------------------------------------
# Permutation: drives covariance for the pure-permutation modes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["fixed_constant", "individual"])
def test_permutation_matches_vector_conversion(mode: str) -> None:
    v = _physics_first_vec(mode, N_PHYSICS, N_PHI)
    perm = scaling_first_permutation(n_physics=N_PHYSICS, mode=mode, n_phi=N_PHI)
    sf_via_perm = v[perm]
    sf_via_fn = physics_first_to_scaling_first(v, n_physics=N_PHYSICS, mode=mode, n_phi=N_PHI)
    np.testing.assert_array_equal(sf_via_perm, sf_via_fn)
    # perm is a genuine permutation of its range
    assert sorted(perm.tolist()) == list(range(len(v)))


def test_fixed_constant_permutation_is_identity() -> None:
    perm = scaling_first_permutation(n_physics=N_PHYSICS, mode="fixed_constant", n_phi=N_PHI)
    np.testing.assert_array_equal(perm, np.arange(N_PHYSICS))


def test_auto_averaged_permutation_raises() -> None:
    # auto_averaged is broadcast/compress, not a permutation — no index map.
    with pytest.raises(ValueError, match="not a permutation"):
        scaling_first_permutation(n_physics=N_PHYSICS, mode="auto_averaged", n_phi=N_PHI)


# ---------------------------------------------------------------------------
# Covariance permuter
# ---------------------------------------------------------------------------
def _random_spd(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)  # symmetric positive-definite


@pytest.mark.parametrize("mode", ["fixed_constant", "individual"])
def test_permute_cov_own_inverse(mode: str) -> None:
    perm = scaling_first_permutation(n_physics=N_PHYSICS, mode=mode, n_phi=N_PHI)
    n = len(perm)
    P = _random_spd(n)
    inv_perm = np.argsort(perm)
    P2 = permute_cov(P, perm)
    P_back = permute_cov(P2, inv_perm)
    np.testing.assert_array_equal(P_back, P)


@pytest.mark.parametrize("mode", ["fixed_constant", "individual"])
def test_permute_cov_preserves_symmetry(mode: str) -> None:
    perm = scaling_first_permutation(n_physics=N_PHYSICS, mode=mode, n_phi=N_PHI)
    P = _random_spd(len(perm))
    assert np.allclose(P, P.T)
    P2 = permute_cov(P, perm)
    np.testing.assert_array_equal(P2, P2.T)


def test_permute_cov_hand_checked() -> None:
    # 2x2 swap: perm=[1,0] swaps both rows and cols.
    P = np.array([[1.0, 2.0], [2.0, 3.0]])
    swapped = permute_cov(P, np.array([1, 0]))
    np.testing.assert_array_equal(swapped, np.array([[3.0, 2.0], [2.0, 1.0]]))


def test_permute_cov_moves_diagonal_correctly() -> None:
    # Diagonal entries follow the permutation: P2[i,i] == P[perm[i], perm[i]].
    perm = scaling_first_permutation(n_physics=N_PHYSICS, mode="individual", n_phi=N_PHI)
    n = len(perm)
    P = np.diag(np.arange(n, dtype=np.float64))
    P2 = permute_cov(P, perm)
    for i in range(n):
        assert P2[i, i] == P[perm[i], perm[i]]


def test_permute_cov_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="square"):
        permute_cov(np.zeros((3, 4)), np.array([0, 1, 2]))


def test_permute_cov_rejects_mismatched_perm() -> None:
    with pytest.raises(ValueError, match="does not match"):
        permute_cov(np.eye(3), np.array([0, 1]))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_rejects_fourier_mode() -> None:
    with pytest.raises(ValueError, match="not an in-scope"):
        physics_first_to_scaling_first(
            np.zeros(N_PHYSICS), n_physics=N_PHYSICS, mode="fourier", n_phi=N_PHI
        )


def test_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="expected"):
        physics_first_to_scaling_first(
            np.zeros(N_PHYSICS + 99), n_physics=N_PHYSICS, mode="individual", n_phi=N_PHI
        )
