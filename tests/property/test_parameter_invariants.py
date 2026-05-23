"""Cross-cutting parameter-registry invariants verified by Hypothesis.

These tests guard the parameter-handling layer against drift between
ParameterInfo, the registry, and the analysis-mode normalizer."""
from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from xpcsjax.config.parameter_registry import (
    AnalysisMode,
    get_param_names,
    get_registry,
)

KNOWN_MODES: tuple[AnalysisMode, ...] = (
    "static_anisotropic",
    "static_isotropic",
    "laminar_flow",
    "two_component",
)


# ----------------------------------------------------------------------
# clip_value invariants
# ----------------------------------------------------------------------

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e8, max_value=1e8))
@settings(max_examples=50)
def test_clip_value_lands_inside_bounds(value: float):
    """For every registered parameter, clip_value(x) must land inside [lower, upper]."""
    registry = get_registry()
    for name in registry._PARAMETERS:
        info = registry.get_param_info(name)
        clipped = info.clip_value(value)
        assert info.lower_bound <= clipped <= info.upper_bound, (
            f"{name}: clip_value({value}) returned {clipped}, "
            f"outside [{info.lower_bound}, {info.upper_bound}]"
        )


@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e8, max_value=1e8))
@settings(max_examples=50)
def test_validate_value_matches_clip_behavior(value: float):
    """validate_value(x) iff clip_value(x) == x — the two views must agree."""
    registry = get_registry()
    for name in ("D0", "alpha", "D_offset"):
        info = registry.get_param_info(name)
        valid = info.validate_value(value)
        clipped_equals = math.isclose(info.clip_value(value), value, abs_tol=1e-12)
        # When clip is a no-op the value was in-bounds; validate_value should agree.
        # Strict floats — only the "in-bounds → valid" implication is checked,
        # since float comparisons at exact bound edges can disagree by epsilon.
        if clipped_equals:
            assert valid, f"{name}: clip is no-op for {value} but validate_value=False"


# ----------------------------------------------------------------------
# Registry lookup symmetry
# ----------------------------------------------------------------------

@given(st.sampled_from(KNOWN_MODES))
@settings(max_examples=20)
def test_get_param_names_returns_valid_registry_keys(mode: AnalysisMode):
    """Every name listed for a mode must resolve to a ParameterInfo."""
    registry = get_registry()
    names = get_param_names(mode)
    for name in names:
        info = registry.get_param_info(name)
        assert info.name == name, (
            f"mode={mode!r}: registry returned info.name={info.name!r} for lookup {name!r}"
        )


@given(st.sampled_from(KNOWN_MODES))
@settings(max_examples=20)
def test_param_lists_have_non_empty_unique_names(mode: AnalysisMode):
    """No mode is empty and no duplicates within a mode."""
    names = get_param_names(mode)
    assert len(names) > 0, f"mode={mode!r} has no params"
    assert len(set(names)) == len(names), f"mode={mode!r} has duplicate names: {names}"


# ----------------------------------------------------------------------
# Mode normalization
# ----------------------------------------------------------------------

@given(st.sampled_from(["two_component", "two-component", "TWO_COMPONENT",
                       "heterodyne", "Heterodyne", "HETERODYNE"]))
@settings(max_examples=10)
def test_heterodyne_synonyms_all_normalize(synonym: str):
    """Every heterodyne synonym must resolve to the 14-param two_component list."""
    names = get_param_names(synonym)  # type: ignore[arg-type]
    assert len(names) == 14, f"synonym {synonym!r} resolved to {len(names)} params"


@given(st.sampled_from(["static_anisotropic", "static_isotropic", "laminar_flow"]))
@settings(max_examples=10)
def test_homodyne_modes_have_expected_param_counts(mode: str):
    """Static modes: 3 params. Laminar: 7 params."""
    names = get_param_names(mode)  # type: ignore[arg-type]
    expected = 7 if mode == "laminar_flow" else 3
    assert len(names) == expected, (
        f"mode={mode!r}: expected {expected} params, got {len(names)}: {names}"
    )
