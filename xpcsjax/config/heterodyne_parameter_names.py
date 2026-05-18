"""Parameter name constants for 14-parameter heterodyne model.

The heterodyne model describes two-component correlation with:
- Reference component: diffusive transport (D0_ref, alpha_ref, D_offset_ref)
- Sample component: diffusive transport (D0_sample, alpha_sample, D_offset_sample)
- Velocity field: time-dependent flow (v0, beta, v_offset)
- Fraction: sample fraction evolution (f0, f1, f2, f3)
- Angle: flow angle relative to q-vector (phi0)
"""

from __future__ import annotations

# Reference transport parameters: J_r(t) = D0_ref * t^alpha_ref + D_offset_ref
REFERENCE_PARAMS: tuple[str, ...] = ("D0_ref", "alpha_ref", "D_offset_ref")

# Sample transport parameters: J_s(t) = D0_sample * t^alpha_sample + D_offset_sample
SAMPLE_PARAMS: tuple[str, ...] = ("D0_sample", "alpha_sample", "D_offset_sample")

# Velocity parameters: v(t) = v0 * t^beta + v_offset
VELOCITY_PARAMS: tuple[str, ...] = ("v0", "beta", "v_offset")

# Fraction parameters: f_s(t) = f0 * exp(f1 * (t - f2)) + f3
FRACTION_PARAMS: tuple[str, ...] = ("f0", "f1", "f2", "f3")

# Angle parameter: flow angle relative to scattering vector
ANGLE_PARAMS: tuple[str, ...] = ("phi0",)

# Scaling parameters: speckle contrast and baseline offset
# These follow the homodyne convention (c2 = offset + contrast × [...])
# and are tracked in the parameter space but NOT in the 14-element
# physics parameter array passed to the JIT backend.
SCALING_PARAMS: tuple[str, ...] = ("contrast", "offset")

# All 14 parameter names in canonical order
ALL_PARAM_NAMES: tuple[str, ...] = (
    # Reference transport (3)
    "D0_ref",
    "alpha_ref",
    "D_offset_ref",
    # Sample transport (3)
    "D0_sample",
    "alpha_sample",
    "D_offset_sample",
    # Velocity (3)
    "v0",
    "beta",
    "v_offset",
    # Fraction (4)
    "f0",
    "f1",
    "f2",
    "f3",
    # Angle (1)
    "phi0",
)

# Parameter groups for organized access
PARAM_GROUPS: dict[str, tuple[str, ...]] = {
    "reference": REFERENCE_PARAMS,
    "sample": SAMPLE_PARAMS,
    "velocity": VELOCITY_PARAMS,
    "fraction": FRACTION_PARAMS,
    "angle": ANGLE_PARAMS,
    "scaling": SCALING_PARAMS,
}

# All parameter names including scaling (16 total)
ALL_PARAM_NAMES_WITH_SCALING: tuple[str, ...] = ALL_PARAM_NAMES + SCALING_PARAMS

# Parameter indices in flattened array
PARAM_INDICES: dict[str, int] = {name: i for i, name in enumerate(ALL_PARAM_NAMES)}


def get_param_index(name: str) -> int:
    """Get index of parameter in flattened array.

    Args:
        name: Parameter name

    Returns:
        Index (0-13)

    Raises:
        KeyError: If parameter name is invalid
    """
    if name not in PARAM_INDICES:
        valid = ", ".join(ALL_PARAM_NAMES)
        raise KeyError(f"Unknown parameter '{name}'. Valid names: {valid}")
    return PARAM_INDICES[name]


def get_group_indices(group: str) -> tuple[int, ...]:
    """Get indices for all parameters in a group.

    Args:
        group: Group name ('reference', 'sample', 'velocity', 'fraction', 'angle')

    Returns:
        Tuple of indices
    """
    if group not in PARAM_GROUPS:
        valid = ", ".join(PARAM_GROUPS.keys())
        raise KeyError(f"Unknown group '{group}'. Valid groups: {valid}")
    return tuple(PARAM_INDICES[name] for name in PARAM_GROUPS[group])
