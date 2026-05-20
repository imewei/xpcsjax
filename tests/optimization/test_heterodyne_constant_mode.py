"""Tests for true `constant` mode in heterodyne (quantile-frozen scaling)."""
from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.core.heterodyne_scaling_utils import (
    estimate_per_angle_scaling_from_quantile,
)


def _make_synthetic_c2(n_phi: int = 3, n_t: int = 32, seed: int = 0) -> dict:
    """Build a tiny synthetic two-time correlation stack for unit tests."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_t)
    t1, t2 = np.meshgrid(t, t, indexing="ij")
    # Toy heterodyne forward: c2 = offset + contrast * exp(-D * |t1-t2|)
    true_contrast = np.array([0.45, 0.42, 0.40][:n_phi])
    true_offset = np.array([1.00, 1.00, 1.00][:n_phi])
    # D = 10 over t in [0, 1] gives max-lag decay = exp(-10) ~ 5e-5, so the
    # large-lag corner c2 ~ offset. Lower D leaves residual signal in the
    # corners and the dual-region offset estimate cannot reach the true value.
    D = 10.0
    decay = np.exp(-D * np.abs(t1 - t2))
    c2 = np.stack(
        [true_offset[i] + true_contrast[i] * decay for i in range(n_phi)], axis=0
    )
    c2 += 0.005 * rng.standard_normal(c2.shape)
    return {
        "c2": c2,
        "t1": np.broadcast_to(t1, c2.shape).copy(),
        "t2": np.broadcast_to(t2, c2.shape).copy(),
        "phi_indices": np.repeat(np.arange(n_phi), n_t * n_t).reshape(c2.shape),
        "true_contrast": true_contrast,
        "true_offset": true_offset,
    }


def test_quantile_estimator_recovers_synthetic_contrast() -> None:
    """Dual-region quantile estimator recovers per-angle contrast and offset.

    Small-lag high-quantile gives the ceiling (offset + contrast); large-lag
    low-quantile gives the floor (offset after decay). Their difference
    recovers contrast.
    """
    data = _make_synthetic_c2(n_phi=3)
    contrast_hat, offset_hat = estimate_per_angle_scaling_from_quantile(
        c2_data=data["c2"],
        t1=data["t1"],
        t2=data["t2"],
        phi_indices=data["phi_indices"],
        n_phi=3,
        quantile=0.95,
    )

    assert contrast_hat.shape == (3,)
    assert offset_hat.shape == (3,)
    # Tolerance reflects synthetic noise level (sigma=0.005, contrast~0.4).
    np.testing.assert_allclose(contrast_hat, data["true_contrast"], rtol=0.10)
    np.testing.assert_allclose(offset_hat, data["true_offset"], rtol=0.02)


def test_quantile_estimator_raises_on_empty_phi_cell() -> None:
    """A phi index with no samples in `phi_indices` is a malformed input."""
    data = _make_synthetic_c2(n_phi=2)
    # phi_indices claims n_phi=2 but the array assigns all samples to index 0
    bad_phi_indices = np.zeros_like(data["phi_indices"])
    with pytest.raises(
        ValueError, match=r"no samples for phi index 1|only.*finite samples"
    ):
        estimate_per_angle_scaling_from_quantile(
            c2_data=data["c2"],
            t1=data["t1"],
            t2=data["t2"],
            phi_indices=bad_phi_indices,
            n_phi=2,
        )


def test_quantile_estimator_diagonal_only_fails_to_recover_offset() -> None:
    """Locks in the dual-region rationale: diagonal-only input cannot recover offset.

    If a future contributor optimizes the wrapper to use only diagonal samples,
    this test will fail and force the regression review.
    """
    # Build a fixture where t1 == t2 everywhere — i.e., only diagonal samples.
    # The dual-region estimator's large-lag region has no data, so it should
    # either raise (preferred) or return a wildly wrong offset.
    n_phi, n_t = 2, 256
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, n_t)
    # All samples are at t1 = t2 = same point → no off-diagonal coverage
    c2_diag = np.stack(
        [0.45 + 1.0 + 0.005 * rng.standard_normal(n_t) for _ in range(n_phi)]
    )
    t1 = np.broadcast_to(t, c2_diag.shape).copy()
    t2 = t1.copy()
    phi_indices = np.repeat(np.arange(n_phi), n_t).reshape(c2_diag.shape)

    # Expect either ValueError (insufficient samples in large-lag region after
    # the guard fires) or a wildly-wrong offset that fails the rtol=0.02 check.
    try:
        contrast_hat, offset_hat = estimate_per_angle_scaling_from_quantile(
            c2_data=c2_diag,
            t1=t1,
            t2=t2,
            phi_indices=phi_indices,
            n_phi=n_phi,
        )
    except ValueError:
        # Acceptable outcome — guard caught the malformed input.
        return
    # If no exception, the offset estimate must NOT pass the tight tolerance:
    del contrast_hat  # not asserted here; the lock-in is on offset
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(offset_hat, [1.0, 1.0], rtol=0.02)
