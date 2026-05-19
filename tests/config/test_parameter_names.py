"""Smoke tests for the canonical parameter-name constants.

``xpcsjax/config/parameter_names.py`` carries the *single source of truth* for
homodyne parameter names — every solver, result builder, and config validator
imports from here. If the constants drift (typo, accidental rename, length
mismatch with the registry), downstream code silently mis-orders parameters
and the resulting fits become physically nonsensical.

These tests are deliberately minimal: structural assertions on the lists,
not behavioural assertions on the solvers. The point is to fence the
constants module so a refactor that touches it cannot pass CI without
acknowledging the change.
"""
from __future__ import annotations


def test_static_isotropic_has_5_params() -> None:
    """Static isotropic = 2 scaling + 3 physical = 5 names."""
    from xpcsjax.config.parameter_names import STATIC_ISOTROPIC_PARAMS

    assert len(STATIC_ISOTROPIC_PARAMS) == 5
    assert STATIC_ISOTROPIC_PARAMS[:2] == ["contrast", "offset"]
    assert STATIC_ISOTROPIC_PARAMS[2:] == ["D0", "alpha", "D_offset"]


def test_laminar_flow_has_9_params() -> None:
    """Laminar flow = 2 scaling + 3 physical + 4 flow = 9 names."""
    from xpcsjax.config.parameter_names import LAMINAR_FLOW_PARAMS

    assert len(LAMINAR_FLOW_PARAMS) == 9
    # Scaling block first, physical second, flow last — canonical order.
    assert LAMINAR_FLOW_PARAMS[:2] == ["contrast", "offset"]
    assert LAMINAR_FLOW_PARAMS[2:5] == ["D0", "alpha", "D_offset"]
    assert LAMINAR_FLOW_PARAMS[5:] == [
        "gamma_dot_t0",
        "beta",
        "gamma_dot_t_offset",
        "phi0",
    ]


def test_get_parameter_names_dispatches_on_mode() -> None:
    """``get_parameter_names(mode)`` must dispatch to the matching constant
    list. Catches a bug where the dispatcher returns the wrong mode's tuple
    (which would mis-order params 5/6 silently)."""
    from xpcsjax.config.parameter_names import (
        LAMINAR_FLOW_PARAMS,
        STATIC_ISOTROPIC_PARAMS,
        get_parameter_names,
    )

    assert get_parameter_names("static_isotropic") == STATIC_ISOTROPIC_PARAMS
    assert get_parameter_names("static_anisotropic") == STATIC_ISOTROPIC_PARAMS
    assert get_parameter_names("laminar_flow") == LAMINAR_FLOW_PARAMS


def test_scaling_and_physical_blocks_are_disjoint() -> None:
    """No parameter name appears in both the scaling and physical blocks —
    otherwise the parameter manager's per-block bounds resolution
    double-counts. Catches a rename collision."""
    from xpcsjax.config.parameter_names import (
        FLOW_PARAMS,
        SCALING_PARAMS,
        STATIC_PHYSICAL_PARAMS,
    )

    scaling = set(SCALING_PARAMS)
    physical = set(STATIC_PHYSICAL_PARAMS)
    flow = set(FLOW_PARAMS)

    assert scaling.isdisjoint(physical), (
        f"scaling/physical overlap: {scaling & physical}"
    )
    assert scaling.isdisjoint(flow), f"scaling/flow overlap: {scaling & flow}"
    assert physical.isdisjoint(flow), f"physical/flow overlap: {physical & flow}"
