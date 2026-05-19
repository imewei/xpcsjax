"""Smoke tests for the heterodyne physics constants + validators.

``xpcsjax/core/heterodyne_physics.py`` carries the canonical physical
constants (Boltzmann, Planck, q/time ranges) and the parameter-bounds
dictionary that ``ParameterManager`` falls back to when YAML doesn't
override. The ``TransportPhysics`` helpers convert raw fit parameters
into human-readable physical regimes ("subdiffusive" vs "ballistic" etc.)
that the result builders log.

If any of these constants drift the consequence is silent — fits still
*run*, just on different physics. These tests fence the registry so a
refactor must acknowledge a bounds change.
"""
from __future__ import annotations

import numpy as np

from xpcsjax.core.heterodyne_physics import (
    PARAMETER_BOUNDS,
    PhysicsConstants,
    TransportPhysics,
    ValidationResult,
    get_default_bounds_array,
)


def test_validation_result_str_pass_and_fail() -> None:
    """ValidationResult formats human-readable strings for both pass and fail
    states. The result builder uses this directly in log lines, so a
    regression in __str__ would corrupt CI logs."""
    ok = ValidationResult(valid=True, message="ok", parameters_checked=14)
    bad = ValidationResult(
        valid=False,
        message="2 violations",
        violations=["D0_ref out of range", "alpha_sample NaN"],
        parameters_checked=14,
    )

    assert "OK" in str(ok)
    assert "FAIL" in str(bad)
    assert "D0_ref" in str(bad)
    assert "alpha_sample" in str(bad)


def test_physics_constants_are_si_values() -> None:
    """The class-level physical constants match SI exact-value definitions
    (post-2019 redefinition). If anyone "rounds" them, the test catches it."""
    assert PhysicsConstants.k_B == 1.380649e-23
    assert PhysicsConstants.h == 6.62607015e-34
    assert PhysicsConstants.c == 299792458.0

    # XPCS-typical q / t ranges (informational, not exact-defined).
    assert PhysicsConstants.Q_MIN_TYPICAL > 0
    assert PhysicsConstants.Q_MAX_TYPICAL > PhysicsConstants.Q_MIN_TYPICAL
    assert PhysicsConstants.TIME_MIN_XPCS < PhysicsConstants.TIME_MAX_XPCS

    # Numerical-stability constants for the kernel paths.
    assert PhysicsConstants.EPS > 0
    assert PhysicsConstants.MAX_EXP_ARG > 0
    assert PhysicsConstants.MIN_POSITIVE > 0


def test_parameter_bounds_covers_14_heterodyne_params() -> None:
    """Heterodyne canonical order has 14 parameters; PARAMETER_BOUNDS must
    have an entry for each. A missing key means ParameterManager silently
    falls back to a None/0 bound, which downstream solvers consume blindly."""
    expected_groups = {
        # Reference transport
        "D0_ref", "alpha_ref", "D_offset_ref",
        # Sample transport
        "D0_sample", "alpha_sample", "D_offset_sample",
        # Velocity
        "v0", "v_beta", "v_offset",
        # Fraction
        "f0", "f1", "f2", "f3",
        # Angle
        "phi0_het",
    }
    assert set(PARAMETER_BOUNDS.keys()) == expected_groups
    assert len(PARAMETER_BOUNDS) == 14

    # Each bound must be a (lower, upper) pair with lower < upper.
    for name, (lo, hi) in PARAMETER_BOUNDS.items():
        assert lo < hi, f"{name}: bounds collapsed ({lo} >= {hi})"


def test_get_default_bounds_array_shapes_align() -> None:
    """``get_default_bounds_array`` flattens the dict to two parallel
    14-vectors. Mismatched shapes would crash the NLSQ adapter at trust-region
    init."""
    lower, upper = get_default_bounds_array()
    assert lower.shape == (14,)
    assert upper.shape == (14,)
    assert np.all(lower < upper)


def test_transport_physics_alpha_interpretation_branches() -> None:
    """``interpret_alpha`` returns physically meaningful regime labels
    across the full alpha range. Pins each branch."""
    assert "equilibrium" in TransportPhysics.interpret_alpha(0.0).lower()
    assert "subdiffusive" in TransportPhysics.interpret_alpha(0.3).lower()
    assert "normal diffusion" in TransportPhysics.interpret_alpha(1.0).lower()
    assert "superdiffusive" in TransportPhysics.interpret_alpha(1.4).lower()
    assert "ballistic" in TransportPhysics.interpret_alpha(2.0).lower()
    assert "decelerating" in TransportPhysics.interpret_alpha(-0.5).lower()


def test_transport_physics_effective_diffusion_coefficient() -> None:
    """For J(t) = D0 * t^alpha, the effective D at t is D0 * alpha * t^(alpha-1).
    Verify at the canonical alpha=1 case (normal diffusion): D_eff = D0."""
    D0 = 1e4
    D_eff_normal = TransportPhysics.diffusion_coefficient(D0=D0, alpha=1.0, t=1.0)
    assert D_eff_normal == D0

    # Ballistic (alpha=2) at t=1: D_eff = 2 * D0.
    D_eff_ballistic = TransportPhysics.diffusion_coefficient(D0=D0, alpha=2.0, t=1.0)
    assert D_eff_ballistic == 2.0 * D0
