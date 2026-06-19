"""JAX-free config validation + template loading."""

from xpcsjax.service.config import (
    ValidationReport,
    available_modes,
    template_dict,
    validate_config,
)


def test_available_modes_are_the_four_known():
    modes = set(available_modes())
    assert modes == {"static_isotropic", "static_anisotropic", "laminar_flow", "two_component"}


# Configs use the template schema: initial_parameters.{parameter_names, values}.
def _ip(names, values):
    return {"initial_parameters": {"parameter_names": names, "values": values}}


def test_valid_config_passes():
    cfg = {
        "analysis_mode": "static_isotropic",
        **_ip(["D0", "alpha", "D_offset"], [1000.0, -1.2, 0.0]),
    }
    rep = validate_config(cfg)
    assert isinstance(rep, ValidationReport)
    assert rep.ok and rep.errors == []


def test_unknown_mode_is_an_error():
    rep = validate_config({"analysis_mode": "nope", **_ip([], None)})
    assert not rep.ok
    assert any("mode" in e.lower() for e in rep.errors)


def test_out_of_bounds_value_is_an_error():
    # D0 (diffusion) must be positive; a negative value is out of registry bounds.
    cfg = {
        "analysis_mode": "static_isotropic",
        **_ip(["D0", "alpha", "D_offset"], [-5.0, -1.2, 0.0]),
    }
    rep = validate_config(cfg)
    assert not rep.ok
    assert any("D0" in e for e in rep.errors)


def test_parameter_not_used_by_mode_is_warned():
    # two_component uses v_beta, not beta -> a warning (not a hard error).
    rep = validate_config({"analysis_mode": "two_component", **_ip(["beta"], [1.0])})
    assert any("beta" in w for w in rep.warnings)


def test_values_length_mismatch_is_an_error():
    cfg = {"analysis_mode": "static_isotropic", **_ip(["D0", "alpha", "D_offset"], [1.0])}
    assert not validate_config(cfg).ok


def test_template_dict_loads_a_mode_template():
    tpl = template_dict("laminar_flow")
    assert isinstance(tpl, dict)
    assert "initial_parameters" in tpl
