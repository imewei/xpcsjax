"""Red tests for Codex review round 1, shard core (claims core-B1..B10)."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import xpcsjax
from xpcsjax import HeterodyneModel, HomodyneModel


def _het_model(n=4):
    return HeterodyneModel.from_config(
        {"temporal": {"dt": 0.1, "time_length": n}, "scattering": {"wavevector_q": 0.01}}
    )


# core-B1 reclassified DESIGN by the lead: sync_time_axis cannot distinguish a file shorter than
# end_frame (trim the tail) from the pipeline's frame-0 exclusion (trim the head); the fix needs a
# maintainer choice (reject end_frame beyond the file, or propagate the clamp). Test removed.


# core-B3 reclassified DESIGN by the lead: time_grid wiring is a documented maintainer deferral
# (core/models.py compute_g1_batch docstring, "deferred - see PR #34"); test removed.


def test_core_b5():
    from xpcsjax.core.jax_backend import compute_chi_squared

    t = jnp.arange(3.0)
    data = jnp.ones((3, 3)).at[0, 0].set(jnp.nan)
    sigma = jnp.ones_like(data).at[0, 0].set(0.0)

    def chi2(obs):
        return compute_chi_squared(
            jnp.array([100.0, 0.0, 0.0]),
            obs,
            sigma,
            t,
            t,
            jnp.array([0.0]),
            0.01,
            1e6,
            0.5,
            1.0,
            1.0,
        )

    assert jnp.isclose(chi2(data), chi2(data.at[0, 0].set(1.0)))


def test_core_b6():
    path = Path(xpcsjax.__file__).parent / "config/templates/xpcsjax_static_isotropic.yaml"
    model = HomodyneModel(yaml.safe_load(path.read_text()))
    assert (model.analysis_mode, model.model.n_params) == ("static_isotropic", 3)


def test_core_b7():
    with pytest.raises(ValueError):
        _het_model().compute_residuals(np.ones((4, 1)))


def test_core_b8():
    from xpcsjax.core.diagonal_correction import apply_diagonal_correction

    c2 = jnp.array([[0, 1], [2, 0]])
    assert jnp.all(jnp.diag(apply_diagonal_correction(c2)) == 1.5)


def test_core_b10():
    from xpcsjax.core.physics import validate_parameters_detailed

    result = validate_parameters_detailed(np.array([[999.0, 999.0]]), [(0.0, 1.0)])
    assert not result.valid
