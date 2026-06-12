"""Phase-0 unit tests for PerAngleScalingPlan (spec §4 Seam 3).

quantile_scaling = (contrast_per_angle[n_phi], offset_per_angle[n_phi]) as produced by
compute_quantile_per_angle_scaling. The plan centralizes seed/expand bookkeeping;
residuals stay model-specific.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.per_angle_mode import PerAngleScalingPlan


def _quantiles(n_phi):
    rng = np.random.default_rng(0)
    contrast = rng.uniform(0.1, 0.5, size=n_phi)
    offset = rng.uniform(0.9, 1.1, size=n_phi)
    return contrast, offset


# ---- freeze flag --------------------------------------------------------------

@pytest.mark.parametrize(
    ("mode", "expected"),
    [("constant", True), ("averaged", False), ("individual", False)],
)
def test_freeze_flag(mode, expected):
    c, o = _quantiles(5)
    plan = PerAngleScalingPlan(mode=mode, n_phi=5, n_physics=7, quantile_scaling=(c, o))
    assert plan.freeze is expected


# ---- seed_tail ----------------------------------------------------------------

def test_seed_tail_constant_is_empty():
    c, o = _quantiles(5)
    plan = PerAngleScalingPlan(mode="constant", n_phi=5, n_physics=7, quantile_scaling=(c, o))
    seed = plan.seed_tail()
    assert seed.shape == (0,)


def test_seed_tail_averaged_is_two_means():
    c, o = _quantiles(5)
    plan = PerAngleScalingPlan(mode="averaged", n_phi=5, n_physics=7, quantile_scaling=(c, o))
    seed = plan.seed_tail()
    assert seed.shape == (2,)
    np.testing.assert_allclose(seed, [c.mean(), o.mean()])


def test_seed_tail_individual_is_concat_per_angle():
    c, o = _quantiles(5)
    plan = PerAngleScalingPlan(mode="individual", n_phi=5, n_physics=7, quantile_scaling=(c, o))
    seed = plan.seed_tail()
    assert seed.shape == (10,)
    np.testing.assert_allclose(seed[:5], c)
    np.testing.assert_allclose(seed[5:], o)


# ---- expand_tail round-trips --------------------------------------------------

def test_expand_tail_individual_is_identity():
    c, o = _quantiles(5)
    plan = PerAngleScalingPlan(mode="individual", n_phi=5, n_physics=7, quantile_scaling=(c, o))
    tail = plan.seed_tail()
    contrast, offset = plan.expand_tail(tail)
    assert contrast.shape == (5,)
    assert offset.shape == (5,)
    np.testing.assert_allclose(contrast, c)
    np.testing.assert_allclose(offset, o)


def test_expand_tail_averaged_broadcasts_two_scalars():
    c, o = _quantiles(5)
    plan = PerAngleScalingPlan(mode="averaged", n_phi=5, n_physics=7, quantile_scaling=(c, o))
    contrast, offset = plan.expand_tail(np.array([0.3, 1.05]))
    assert contrast.shape == (5,)
    assert offset.shape == (5,)
    np.testing.assert_allclose(contrast, np.full(5, 0.3))
    np.testing.assert_allclose(offset, np.full(5, 1.05))


def test_expand_tail_constant_broadcasts_frozen_quantiles():
    c, o = _quantiles(5)
    plan = PerAngleScalingPlan(mode="constant", n_phi=5, n_physics=7, quantile_scaling=(c, o))
    # constant ignores the (empty) tail and broadcasts the frozen per-angle quantiles.
    contrast, offset = plan.expand_tail(np.empty(0))
    np.testing.assert_allclose(contrast, c)
    np.testing.assert_allclose(offset, o)


def test_expand_tail_individual_rejects_wrong_length():
    c, o = _quantiles(5)
    plan = PerAngleScalingPlan(mode="individual", n_phi=5, n_physics=7, quantile_scaling=(c, o))
    with pytest.raises(ValueError, match="expected scaling tail of length"):
        plan.expand_tail(np.zeros(7))


# ---- expand_back (full per-angle popt) ----------------------------------------

def test_expand_back_individual_reorders_to_full_per_angle():
    # popt is scaling-first [c0..c4, o0..o4, physics(7)]; expand_back returns the
    # dense per-angle (contrast[n_phi], offset[n_phi], physics) split.
    c, o = _quantiles(5)
    plan = PerAngleScalingPlan(mode="individual", n_phi=5, n_physics=7, quantile_scaling=(c, o))
    physics = np.arange(7.0) + 100.0
    popt = np.concatenate([c, o, physics])
    contrast, offset, phys = plan.expand_back(popt)
    np.testing.assert_allclose(contrast, c)
    np.testing.assert_allclose(offset, o)
    np.testing.assert_allclose(phys, physics)


def test_expand_back_averaged_broadcasts_then_splits_physics():
    c, o = _quantiles(5)
    plan = PerAngleScalingPlan(mode="averaged", n_phi=5, n_physics=7, quantile_scaling=(c, o))
    physics = np.arange(7.0)
    popt = np.concatenate([[0.25, 1.02], physics])
    contrast, offset, phys = plan.expand_back(popt)
    np.testing.assert_allclose(contrast, np.full(5, 0.25))
    np.testing.assert_allclose(offset, np.full(5, 1.02))
    np.testing.assert_allclose(phys, physics)


def test_expand_back_constant_uses_frozen_quantiles_physics_only_vector():
    c, o = _quantiles(5)
    plan = PerAngleScalingPlan(mode="constant", n_phi=5, n_physics=7, quantile_scaling=(c, o))
    physics = np.arange(7.0)
    popt = physics  # physics-only optimizer vector
    contrast, offset, phys = plan.expand_back(popt)
    np.testing.assert_allclose(contrast, c)
    np.testing.assert_allclose(offset, o)
    np.testing.assert_allclose(phys, physics)


# ---- expand_tail_jax: JIT-safe parity with the NumPy expand_tail --------------

@pytest.mark.parametrize("mode", ["constant", "averaged", "individual"])
def test_expand_tail_jax_matches_numpy(mode):
    """The jnp variant returns the same values as the NumPy one (so the traced
    residual sees identical scaling), and must accept a jnp-traced argument."""
    import jax
    import jax.numpy as jnp

    c, o = _quantiles(5)
    plan = PerAngleScalingPlan(mode=mode, n_phi=5, n_physics=7, quantile_scaling=(c, o))
    tail = plan.seed_tail()
    c_np, o_np = plan.expand_tail(tail)
    # call inside jit to prove no TracerArrayConversionError (the bug this guards)
    c_jx, o_jx = jax.jit(plan.expand_tail_jax)(jnp.asarray(tail, dtype=jnp.float64))
    np.testing.assert_allclose(np.asarray(c_jx), c_np, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(o_jx), o_np, rtol=1e-12)


# ---- expand_covariance: dense scaling-first covariance ------------------------

def test_expand_covariance_individual_is_identity():
    c, o = _quantiles(4)
    plan = PerAngleScalingPlan(mode="individual", n_phi=4, n_physics=7, quantile_scaling=(c, o))
    pcov = np.eye(2 * 4 + 7)
    np.testing.assert_array_equal(plan.expand_covariance(pcov), pcov)


def test_expand_covariance_averaged_replicates_scalar_blocks():
    c, o = _quantiles(4)
    plan = PerAngleScalingPlan(mode="averaged", n_phi=4, n_physics=7, quantile_scaling=(c, o))
    # optimizer covariance: [c_avg, o_avg, *7 physics] -> 9x9
    pcov = np.eye(9)
    pcov[0, 0] = 0.25  # contrast_avg variance
    pcov[1, 1] = 0.16  # offset_avg variance
    dense = plan.expand_covariance(pcov)
    assert dense.shape == (2 * 4 + 7, 2 * 4 + 7)
    # every per-angle contrast diag entry takes the shared variance
    for i in range(4):
        assert dense[i, i] == 0.25
        assert dense[4 + i, 4 + i] == 0.16


def test_expand_covariance_constant_has_zero_scaling_block():
    c, o = _quantiles(4)
    plan = PerAngleScalingPlan(mode="constant", n_phi=4, n_physics=7, quantile_scaling=(c, o))
    pcov = np.eye(7)  # physics-only optimizer covariance
    dense = plan.expand_covariance(pcov)
    assert dense.shape == (2 * 4 + 7, 2 * 4 + 7)
    # frozen scaling rows/cols are zero-variance
    assert np.all(dense[: 2 * 4, : 2 * 4] == 0.0)
    np.testing.assert_array_equal(dense[2 * 4 :, 2 * 4 :], pcov)


def test_expand_covariance_none_passes_through():
    c, o = _quantiles(4)
    plan = PerAngleScalingPlan(mode="averaged", n_phi=4, n_physics=7, quantile_scaling=(c, o))
    assert plan.expand_covariance(None) is None


# ---- group_indices delegates to the mapper ------------------------------------

@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("constant", []),
        ("averaged", [(0, 1), (1, 2)]),
        ("individual", [(0, 5), (5, 10)]),
    ],
)
def test_group_indices_match_canonical_mapper(mode, expected):
    c, o = _quantiles(5)
    plan = PerAngleScalingPlan(mode=mode, n_phi=5, n_physics=7, quantile_scaling=(c, o))
    assert plan.group_indices == expected
