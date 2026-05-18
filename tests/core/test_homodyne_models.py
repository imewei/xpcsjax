"""Smoke tests for HomodyneModel classes (DiffusionModel, CombinedModel).

Adaptations from the plan:
- Attribute is ``parameter_names`` (not ``param_names``) in the homodyne source.
- The 4th shear parameter is ``gamma_dot_t0`` / ``gamma_dot_t_offset`` (with the
  ``_t``), not ``gamma_dot_0`` / ``gamma_dot_offset`` as the plan guessed.
- ``models.py`` exposes ``compute_g1`` / ``compute_g2`` / ``compute_chi_squared``,
  not a ``compute_residual`` method. Residual is built one layer up (NLSQ
  adapter, ported in later tasks). The smoke test here therefore drives
  ``compute_g1`` directly with a small synthetic grid.
- ``models.py`` imports ``xpcsjax.core.jax_backend`` / ``model_mixins`` /
  ``physics_utils``. Those modules are ported in later tasks (Task 14+).
  Until then, the import will raise ``ModuleNotFoundError`` and the whole
  test module is skipped at collection time via ``pytest.importorskip``.
"""

from __future__ import annotations

import pytest

# Defer the heavy import behind importorskip so this module is a no-op (rather
# than a collection error) while upstream backend modules are still being
# ported. Once ``xpcsjax.core.jax_backend`` lands, these tests activate.
models = pytest.importorskip(
    "xpcsjax.core.models",
    reason="xpcsjax.core.models depends on jax_backend / model_mixins / "
    "physics_utils which are ported in later NLSQ-merge tasks.",
)

import jax.numpy as jnp  # noqa: E402

DiffusionModel = models.DiffusionModel
CombinedModel = models.CombinedModel


def test_diffusion_model_param_count() -> None:
    """Static diffusion mode has 3 parameters with the expected names."""
    model = DiffusionModel()
    assert len(model.parameter_names) == 3
    assert set(model.parameter_names) == {"D0", "alpha", "D_offset"}


def test_combined_model_param_count() -> None:
    """Laminar flow mode has 7 parameters with the expected names."""
    model = CombinedModel()  # defaults to "laminar_flow"
    assert len(model.parameter_names) == 7
    expected = {
        "D0",
        "alpha",
        "D_offset",
        "gamma_dot_t0",
        "beta",
        "gamma_dot_t_offset",
        "phi0",
    }
    assert set(model.parameter_names) == expected


def test_combined_model_static_mode_param_count() -> None:
    """Static analysis mode collapses to the 3 diffusion parameters."""
    model = CombinedModel(analysis_mode="static")
    assert len(model.parameter_names) == 3
    assert set(model.parameter_names) == {"D0", "alpha", "D_offset"}


def test_diffusion_model_g1_runs() -> None:
    """compute_g1 must return a finite array for in-bounds default params.

    Smoke check for the field correlation g1_diff. We use the model's own
    default parameters (guaranteed to be in-bounds) and a small synthetic
    time grid.
    """
    model = DiffusionModel()
    params = model.get_default_parameters()
    N = 8
    t1 = jnp.arange(N, dtype=jnp.float64)
    t2 = jnp.arange(N, dtype=jnp.float64)
    phi = jnp.array([0.0])
    q = 0.01
    L = 1.0
    dt = 1.0

    g1 = model.compute_g1(params, t1, t2, phi, q, L, dt)
    assert jnp.all(jnp.isfinite(g1))
