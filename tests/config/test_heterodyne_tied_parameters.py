"""Tests for heterodyne tied_parameters (equality-constrained physics params).

See docs/superpowers/specs/2026-07-29-heterodyne-tied-parameters-design.md.
"""

from __future__ import annotations

import pytest

from xpcsjax.config.heterodyne_parameter_space import ParameterSpace


def _base_config(**overrides):
    config = {"analysis_mode": "two_component"}
    config.update(overrides)
    return config


def test_tied_parameters_absent_is_noop():
    space = ParameterSpace.from_config(_base_config())
    assert space.tied == {}
    assert space.vary["D0_ref"] is True


def test_tied_parameters_forces_child_non_varying():
    config = _base_config(initial_parameters={"tied_parameters": {"D0_ref": "D0_sample"}})
    space = ParameterSpace.from_config(config)
    assert space.tied == {"D0_ref": "D0_sample"}
    assert space.vary["D0_ref"] is False
    assert space.vary["D0_sample"] is True


def test_tied_parameters_self_tie_rejected():
    config = _base_config(initial_parameters={"tied_parameters": {"D0_ref": "D0_ref"}})
    with pytest.raises(ValueError, match="cannot be tied to itself"):
        ParameterSpace.from_config(config)


def test_tied_parameters_alias_canonical_collision_rejected():
    # "beta" and its public alias "v_beta" both canonicalize to the same
    # internal child name -- using both as separate tied_parameters keys
    # must be rejected, not silently collapsed to whichever entry the dict
    # comprehension processed last (three-brain audit finding, 2026-08-24).
    config = _base_config(
        initial_parameters={"tied_parameters": {"beta": "v_beta", "v_beta": "D0_ref"}}
    )
    with pytest.raises(ValueError, match="alias and its canonical name"):
        ParameterSpace.from_config(config)


def test_tied_parameters_alias_canonical_collision_reverse_order_rejected():
    # Same collision as above with raw-key insertion order reversed -- the
    # guard must not depend on which key comes first (review-pr
    # pr-test-analyzer finding).
    config = _base_config(
        initial_parameters={"tied_parameters": {"v_beta": "D0_ref", "beta": "v_beta"}}
    )
    with pytest.raises(ValueError, match="alias and its canonical name"):
        ParameterSpace.from_config(config)


def test_tied_parameters_phi0_alias_canonical_collision_rejected():
    # Same collision class on the OTHER alias pair (phi0_het -> phi0), not
    # just v_beta/beta -- confirms the guard is generic over
    # _INBOUND_NAME_ALIAS, not coupled to one specific entry
    # (review-pr pr-test-analyzer finding).
    config = _base_config(
        initial_parameters={"tied_parameters": {"phi0": "phi0_het", "phi0_het": "D0_ref"}}
    )
    with pytest.raises(ValueError, match="alias and its canonical name"):
        ParameterSpace.from_config(config)


def test_tied_parameters_lone_alias_key_still_canonicalizes():
    # The valid, non-colliding case: a single alias used once must still
    # translate to its canonical name (review-pr pr-test-analyzer finding --
    # the collision guard must not fire on ordinary alias usage).
    config = _base_config(initial_parameters={"tied_parameters": {"v_beta": "D0_sample"}})
    space = ParameterSpace.from_config(config)
    assert space.tied == {"beta": "D0_sample"}


def test_tied_parameters_chain_rejected():
    config = _base_config(
        initial_parameters={
            "tied_parameters": {"D0_ref": "D0_sample", "D0_sample": "v0"},
        }
    )
    with pytest.raises(ValueError, match="chained ties are not supported"):
        ParameterSpace.from_config(config)


def test_tied_parameters_unknown_child_rejected():
    config = _base_config(initial_parameters={"tied_parameters": {"not_a_param": "D0_sample"}})
    with pytest.raises(ValueError, match="unknown physics parameter"):
        ParameterSpace.from_config(config)


def test_tied_parameters_unknown_parent_rejected():
    config = _base_config(initial_parameters={"tied_parameters": {"D0_ref": "not_a_param"}})
    with pytest.raises(ValueError, match="unknown physics parameter"):
        ParameterSpace.from_config(config)


def test_tied_parameters_fixed_parent_rejected():
    config = _base_config(
        initial_parameters={
            "parameter_names": ["D0_sample"],
            "values": [123.0],
            "active_parameters": [],  # empty list = fix everything
            "tied_parameters": {"D0_ref": "D0_sample"},
        }
    )
    with pytest.raises(ValueError, match="is not varying"):
        ParameterSpace.from_config(config)


def test_tied_parameters_not_a_mapping_rejected():
    config = _base_config(initial_parameters={"tied_parameters": ["D0_ref", "D0_sample"]})
    with pytest.raises(ValueError, match="must be a mapping"):
        ParameterSpace.from_config(config)


def test_tied_parameters_non_string_value_rejected():
    """A malformed mapping VALUE (unhashable, e.g. an empty list) must raise
    the documented config-load ValueError, not a raw TypeError from the
    internal name-translation dict comprehension (codex/agy audit finding
    #5, 2026-07-29)."""
    config = _base_config(initial_parameters={"tied_parameters": {"D0_ref": []}})
    with pytest.raises(ValueError, match="tied_parameters"):
        ParameterSpace.from_config(config)


def test_tied_parameters_non_string_key_rejected():
    config = _base_config(initial_parameters={"tied_parameters": {("D0_ref",): "D0_sample"}})
    with pytest.raises(ValueError, match="tied_parameters"):
        ParameterSpace.from_config(config)


def test_tied_parameters_no_warning_without_active_parameters(caplog):
    """The 'also listed as varying' warning must be scoped to an EXPLICIT
    active_parameters conflict, not fire on the ordinary (no
    active_parameters) usage pattern where every untouched parameter
    defaults to vary=True (codex/agy audit finding #7, 2026-07-29)."""
    config = _base_config(initial_parameters={"tied_parameters": {"D0_ref": "D0_sample"}})
    with caplog.at_level("WARNING"):
        ParameterSpace.from_config(config)
    assert not any("also listed as varying" in rec.message for rec in caplog.records)


def test_tied_parameters_syncs_child_value_to_parent():
    config = _base_config(
        initial_parameters={
            "parameter_names": ["D0_ref", "D0_sample"],
            "values": [1.0, 5.0],
            "tied_parameters": {"D0_ref": "D0_sample"},
        }
    )
    space = ParameterSpace.from_config(config)
    assert space.values["D0_ref"] == pytest.approx(5.0)


def test_tied_parameters_survives_grouped_format_vary_override():
    """A grouped-format vary: true for a tied child must NOT undo the tie --
    _apply_tied_parameters must run LAST in from_config."""
    config = _base_config(
        initial_parameters={"tied_parameters": {"D0_ref": "D0_sample"}},
        parameters={"reference": {"D0_ref": {"vary": True}}},
    )
    space = ParameterSpace.from_config(config)
    assert space.vary["D0_ref"] is False


def test_tied_parameters_active_parameters_conflict_logs_warning(caplog):
    """child also listed in active_parameters: tie wins, warning logged, not
    a hard error."""
    config = _base_config(
        initial_parameters={
            "active_parameters": ["D0_ref", "D0_sample"],
            "tied_parameters": {"D0_ref": "D0_sample"},
        }
    )
    with caplog.at_level("WARNING"):
        space = ParameterSpace.from_config(config)
    assert space.vary["D0_ref"] is False
    assert any("D0_ref" in rec.message for rec in caplog.records)


def test_tied_parameters_bounds_divergence_logs_warning(caplog):
    """child's configured bounds differing from its tied parent's bounds
    must log a warning, mirroring the existing value-divergence warning.

    NOTE: uses the same bounds-override config surface exercised by
    tests/config/test_heterodyne_config_bounds_override.py -- confirm the
    exact key/shape there (this plan assumes a min/max pair) before writing
    this test; adjust if it differs.
    """
    config = _base_config(
        initial_parameters={"tied_parameters": {"D0_ref": "D0_sample"}},
        parameter_space={
            "bounds": [
                {"name": "D0_ref", "min": 1.0, "max": 100.0},
                {"name": "D0_sample", "min": 1.0, "max": 100000.0},
            ]
        },
    )
    with caplog.at_level("WARNING"):
        ParameterSpace.from_config(config)
    assert any("bound" in rec.message.lower() and "D0_ref" in rec.message for rec in caplog.records)


def test_to_config_round_trips_tied_parameters():
    config = _base_config(initial_parameters={"tied_parameters": {"D0_ref": "D0_sample"}})
    space = ParameterSpace.from_config(config)
    round_tripped = space.to_config()
    assert round_tripped["initial_parameters"]["tied_parameters"] == {"D0_ref": "D0_sample"}


def test_to_config_empty_tied_parameters_when_untied():
    space = ParameterSpace.from_config(_base_config())
    round_tripped = space.to_config()
    assert round_tripped["initial_parameters"]["tied_parameters"] == {}
