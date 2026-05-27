"""HeterodyneModel implements the PhysicsModelBase contract with 14 physics params."""

from __future__ import annotations

import jax.numpy as jnp  # noqa: F401  (imported to ensure JAX is initialized before model use)
import numpy as np  # noqa: F401

from xpcsjax.core.heterodyne_model import HeterodyneModel
from xpcsjax.core.models import PhysicsModelBase


def test_is_physics_model() -> None:
    """HeterodyneModel must satisfy the PhysicsModelBase contract."""
    model = HeterodyneModel()
    assert isinstance(model, PhysicsModelBase)


def test_analysis_mode_is_two_component() -> None:
    """The wrapper reports its analysis_mode."""
    model = HeterodyneModel()
    assert model.analysis_mode == "two_component"


def test_param_names_match_registry() -> None:
    """The 14 heterodyne param names come from the registry, in registry order."""
    model = HeterodyneModel()
    expected = (
        "D0_ref",
        "alpha_ref",
        "D_offset_ref",
        "D0_sample",
        "alpha_sample",
        "D_offset_sample",
        "v0",
        "v_beta",
        "v_offset",
        "f0",
        "f1",
        "f2",
        "f3",
        "phi0_het",
    )
    # Adapt to whatever attribute the base class uses (parameter_names vs param_names)
    names = getattr(model, "parameter_names", None) or getattr(
        model, "param_names", None
    )
    assert names is not None, (
        "HeterodyneModel must expose parameter_names or param_names"
    )
    assert tuple(names) == expected, f"got {tuple(names)}"


def test_param_bounds_match_docs() -> None:
    """Spot-check three bounds against heterodyne docs/registry."""
    model = HeterodyneModel()
    names = getattr(model, "parameter_names", None) or getattr(
        model, "param_names", None
    )
    # Try get_parameter_bounds() method (PhysicsModelBase contract) first
    if hasattr(model, "get_parameter_bounds"):
        bounds = model.get_parameter_bounds()
    else:
        bounds = getattr(model, "parameter_bounds", None) or getattr(
            model, "param_bounds", None
        )
    assert bounds is not None
    names_list = list(names)
    assert bounds[names_list.index("D0_ref")] == (0.0, 1e6)
    assert bounds[names_list.index("v_beta")] == (0.0, 2.0)
    assert bounds[names_list.index("phi0_het")] == (-10.0, 10.0)


def test_default_params_length_14() -> None:
    """get_default_parameters returns a 14-element array."""
    model = HeterodyneModel()
    defaults = model.get_default_parameters()
    assert defaults.shape == (14,)


def test_model_constructible_via_from_config() -> None:
    """If HeterodyneModel exposes a from_config classmethod, it must accept a minimal config dict.

    Skip if from_config doesn't exist — the bare constructor is sufficient.
    """
    if hasattr(HeterodyneModel, "from_config"):
        model = HeterodyneModel.from_config({"analysis_mode": "two_component"})
        assert model is not None
