"""Tests for heterodyne post-hoc per-angle view helpers
(reconstruct_per_angle_scaling, per_angle_chi2).
"""
from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.results import OptimizationResult


def test_reconstruct_per_angle_scaling_fourier_mode() -> None:
    """Helper reconstructs per-angle contrast from Fourier coefficients."""
    from xpcsjax.optimization.nlsq.heterodyne_views import (
        reconstruct_per_angle_scaling,
    )

    # Fake result: 3 physics + 5 contrast Fourier (K=2) + 5 offset Fourier
    # Coefficients: contrast_coeffs = [0.4, 0, 0, 0, 0]  (constant 0.4)
    #               offset_coeffs   = [1.0, 0, 0, 0, 0]  (constant 1.0)
    params = np.array([0.5, 0.1, 0.01, 0.4, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    result = OptimizationResult(
        parameters=params,
        uncertainties=np.zeros_like(params),
        covariance=np.eye(len(params)),
        chi_squared=0.0,
        reduced_chi_squared=0.0,
        convergence_status="converged",
        iterations=0,
        execution_time=0.0,
        device_info={},
        recovery_actions=[],
        quality_flag="good",
        streaming_diagnostics=None,
        stratification_diagnostics=None,
        nlsq_diagnostics={
            "per_angle_mode": "fourier",
            "fourier_basis_dim": 5,
            "scaling_source": "fitted",
        },
    )

    phi = np.array([0.0, 45.0, 90.0])
    layout = {"n_physics": 3, "fourier_order": 2}
    out = reconstruct_per_angle_scaling(
        result=result, phi_angles=phi, mode="fourier", layout=layout
    )

    assert set(out.keys()) == {"contrast", "offset"}
    np.testing.assert_allclose(out["contrast"], [0.4, 0.4, 0.4], atol=1e-12)
    np.testing.assert_allclose(out["offset"], [1.0, 1.0, 1.0], atol=1e-12)


def test_reconstruct_per_angle_scaling_fourier_matches_canonical_basis() -> None:
    """Helper's Fourier basis layout matches FourierReparameterizer's.

    Exercises every harmonic with non-zero coefficients at K=2 and compares
    the helper output against B @ coeffs, where B is the canonical basis
    matrix built by `FourierReparameterizer._compute_basis_matrix`. This is
    the load-bearing test for the interleaved [c0, c1, s1, c2, s2] layout.
    """
    from xpcsjax.optimization.nlsq.fourier_reparam import (
        FourierReparamConfig,
        FourierReparameterizer,
    )
    from xpcsjax.optimization.nlsq.heterodyne_views import reconstruct_per_angle_scaling

    K = 2
    phi_deg = np.array([0.0, 30.0, 60.0, 90.0, 135.0])
    n_physics = 3
    basis_dim = 2 * K + 1  # 5

    # Non-zero coefficients to exercise every harmonic.
    contrast_coeffs = np.array([0.40, 0.05, 0.03, -0.02, 0.01])  # [c0, c1, s1, c2, s2]
    offset_coeffs = np.array([1.00, 0.10, -0.05, 0.02, -0.03])

    params = np.concatenate([
        np.array([0.5, 0.1, 0.01]),  # physics
        contrast_coeffs,
        offset_coeffs,
    ])

    result = OptimizationResult(
        parameters=params,
        uncertainties=np.zeros_like(params),
        covariance=np.eye(len(params)),
        chi_squared=0.0,
        reduced_chi_squared=0.0,
        convergence_status="converged",
        iterations=0,
        execution_time=0.0,
        device_info={},
        recovery_actions=[],
        quality_flag="good",
        streaming_diagnostics=None,
        stratification_diagnostics=None,
        nlsq_diagnostics={
            "per_angle_mode": "fourier",
            "fourier_basis_dim": basis_dim,
            "scaling_source": "fitted",
        },
    )

    out = reconstruct_per_angle_scaling(
        result=result,
        phi_angles=phi_deg,
        mode="fourier",
        layout={"n_physics": n_physics, "fourier_order": K},
    )

    # Build the canonical basis matrix the same way `_fit_joint_multi_phi`
    # would. FourierReparameterizer takes phi in *radians*; the helper takes
    # phi in *degrees* and deg2rads internally — so we feed the radian form
    # to the reparameterizer for an apples-to-apples comparison.
    config = FourierReparamConfig(mode="fourier", fourier_order=K)
    reparam = FourierReparameterizer(
        phi_angles=np.deg2rad(phi_deg), config=config
    )
    assert reparam.use_fourier, "expected Fourier mode to be active"
    B = reparam.get_basis_matrix()
    assert B is not None
    expected_contrast = B @ contrast_coeffs
    expected_offset = B @ offset_coeffs

    np.testing.assert_allclose(out["contrast"], expected_contrast, atol=1e-12)
    np.testing.assert_allclose(out["offset"], expected_offset, atol=1e-12)


def test_reconstruct_per_angle_scaling_individual_mode() -> None:
    from xpcsjax.optimization.nlsq.heterodyne_views import reconstruct_per_angle_scaling

    n_physics = 3
    params = np.concatenate([
        np.array([0.5, 0.1, 0.01]),       # physics
        np.array([0.4, 0.42, 0.38]),       # contrast per angle
        np.array([1.0, 1.0, 1.0]),         # offset per angle
    ])
    result = OptimizationResult(
        parameters=params,
        uncertainties=np.zeros_like(params),
        covariance=np.eye(len(params)),
        chi_squared=0.0,
        reduced_chi_squared=0.0,
        convergence_status="converged",
        iterations=0,
        execution_time=0.0,
        device_info={},
        recovery_actions=[],
        quality_flag="good",
        streaming_diagnostics=None,
        stratification_diagnostics=None,
        nlsq_diagnostics={"per_angle_mode": "individual", "scaling_source": "fitted"},
    )

    out = reconstruct_per_angle_scaling(
        result=result,
        phi_angles=np.array([0.0, 45.0, 90.0]),
        mode="individual",
        layout={"n_physics": n_physics},
    )

    np.testing.assert_allclose(out["contrast"], [0.4, 0.42, 0.38])
    np.testing.assert_allclose(out["offset"], [1.0, 1.0, 1.0])


def test_reconstruct_per_angle_scaling_constant_mode() -> None:
    from xpcsjax.optimization.nlsq.heterodyne_views import reconstruct_per_angle_scaling

    result = OptimizationResult(
        parameters=np.array([0.5, 0.1, 0.01]),  # physics only
        uncertainties=np.zeros(3),
        covariance=np.eye(3),
        chi_squared=0.0,
        reduced_chi_squared=0.0,
        convergence_status="converged",
        iterations=0,
        execution_time=0.0,
        device_info={},
        recovery_actions=[],
        quality_flag="good",
        streaming_diagnostics=None,
        stratification_diagnostics=None,
        nlsq_diagnostics={
            "per_angle_mode": "constant",
            "scaling_source": "quantile_fixed",
            "contrast_per_angle_fixed": np.array([0.4, 0.42, 0.38]),
            "offset_per_angle_fixed": np.array([1.0, 1.0, 1.0]),
        },
    )

    out = reconstruct_per_angle_scaling(
        result=result,
        phi_angles=np.array([0.0, 45.0, 90.0]),
        mode="constant",
        layout={"n_physics": 3},
    )

    np.testing.assert_allclose(out["contrast"], [0.4, 0.42, 0.38])
    np.testing.assert_allclose(out["offset"], [1.0, 1.0, 1.0])


def test_reconstruct_per_angle_scaling_auto_mode_resolves_via_diagnostics() -> None:
    """`mode='auto'` reads the actual dispatched mode from diagnostics and recurses."""
    from xpcsjax.optimization.nlsq.heterodyne_views import reconstruct_per_angle_scaling

    # Build a result that came out of `auto` dispatch but landed in `constant`.
    result = OptimizationResult(
        parameters=np.array([0.5, 0.1, 0.01]),
        uncertainties=np.zeros(3),
        covariance=np.eye(3),
        chi_squared=0.0,
        reduced_chi_squared=0.0,
        convergence_status="converged",
        iterations=0,
        execution_time=0.0,
        device_info={},
        recovery_actions=[],
        quality_flag="good",
        streaming_diagnostics=None,
        stratification_diagnostics=None,
        nlsq_diagnostics={
            "per_angle_mode": "constant",  # the resolved mode
            "scaling_source": "quantile_fixed",
            "contrast_per_angle_fixed": np.array([0.4, 0.42, 0.38]),
            "offset_per_angle_fixed": np.array([1.0, 1.0, 1.0]),
        },
    )

    out = reconstruct_per_angle_scaling(
        result=result,
        phi_angles=np.array([0.0, 45.0, 90.0]),
        mode="auto",
        layout={"n_physics": 3},
    )

    np.testing.assert_allclose(out["contrast"], [0.4, 0.42, 0.38])


def test_per_angle_chi2_reads_from_diagnostics() -> None:
    """`per_angle_chi2()` retrieves the array from nlsq_diagnostics."""
    from xpcsjax.optimization.nlsq.heterodyne_views import per_angle_chi2

    expected = np.array([1.0, 2.0, 3.0])
    result = OptimizationResult(
        parameters=np.array([0.5]),
        uncertainties=np.zeros(1),
        covariance=np.eye(1),
        chi_squared=6.0,
        reduced_chi_squared=0.0,
        convergence_status="converged",
        iterations=0,
        execution_time=0.0,
        device_info={},
        recovery_actions=[],
        quality_flag="good",
        streaming_diagnostics=None,
        stratification_diagnostics=None,
        nlsq_diagnostics={"chi2_per_angle": expected},
    )
    np.testing.assert_array_equal(per_angle_chi2(result), expected)


def test_per_angle_chi2_raises_when_missing() -> None:
    """`per_angle_chi2()` raises ValueError if the key isn't present."""
    from xpcsjax.optimization.nlsq.heterodyne_views import per_angle_chi2

    result = OptimizationResult(
        parameters=np.array([0.5]),
        uncertainties=np.zeros(1),
        covariance=np.eye(1),
        chi_squared=0.0,
        reduced_chi_squared=0.0,
        convergence_status="converged",
        iterations=0,
        execution_time=0.0,
        device_info={},
        recovery_actions=[],
        quality_flag="good",
        streaming_diagnostics=None,
        stratification_diagnostics=None,
        nlsq_diagnostics=None,
    )
    with pytest.raises(ValueError, match="chi2_per_angle"):
        per_angle_chi2(result)
