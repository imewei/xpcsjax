"""Tests for xpcsjax.optimization.nlsq.shear_weighting.

The shear weight ``w(phi) = w_min + (1 - w_min) * |cos(phi0 - phi)|^alpha`` is
exactly verifiable: it equals 1 at angles parallel/antiparallel to flow and
``w_min`` at perpendicular angles (unnormalized). Weighted-loss application is
checked against a hand-computed sum, and the disabled path against the plain
(unweighted) MSE.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from xpcsjax.optimization.nlsq import shear_weighting as sw

# ---------------------------------------------------------------------------
# ShearWeightingConfig
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = sw.ShearWeightingConfig()
    assert cfg.enable is True
    assert cfg.min_weight == 0.3
    assert cfg.normalize is True


def test_config_from_config_dict() -> None:
    cfg = sw.ShearWeightingConfig.from_config(
        {
            "shear_weighting": {
                "enable": False,
                "min_weight": 0.5,
                "alpha": 2.0,
                "initial_phi0": 10.0,
            }
        }
    )
    assert cfg.enable is False
    assert cfg.min_weight == 0.5
    assert cfg.alpha == 2.0
    assert cfg.initial_phi0 == 10.0


def test_config_from_config_initial_phi0_none() -> None:
    cfg = sw.ShearWeightingConfig.from_config({"shear_weighting": {}})
    assert cfg.initial_phi0 is None


# ---------------------------------------------------------------------------
# weight values (the physics)
# ---------------------------------------------------------------------------


def _weighter(normalize: bool = False, **kw: object) -> sw.ShearSensitivityWeighting:
    cfg = sw.ShearWeightingConfig(
        min_weight=0.3,
        alpha=1.0,
        normalize=normalize,
        initial_phi0=0.0,
        **kw,  # type: ignore[arg-type]
    )
    return sw.ShearSensitivityWeighting(
        np.array([0.0, 90.0, 180.0, 270.0]), n_physical=7, phi0_index=6, config=cfg
    )


def test_weights_peak_at_parallel_and_floor_at_perpendicular() -> None:
    w = _weighter(normalize=False).get_weights()
    # phi0=0: parallel (0, 180) -> 1.0; perpendicular (90, 270) -> min_weight.
    np.testing.assert_allclose(w[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(w[2], 1.0, atol=1e-6)
    np.testing.assert_allclose(w[1], 0.3, atol=1e-6)
    np.testing.assert_allclose(w[3], 0.3, atol=1e-6)


def test_weights_normalized_to_unit_mean() -> None:
    w = _weighter(normalize=True).get_weights()
    np.testing.assert_allclose(np.mean(w), 1.0, atol=1e-6)


def test_get_weights_override_phi0() -> None:
    weighter = _weighter(normalize=False)
    # Override to phi0=90: now 90/270 are parallel.
    w = weighter.get_weights(phi0_current=90.0)
    np.testing.assert_allclose(w[1], 1.0, atol=1e-6)
    np.testing.assert_allclose(w[0], 0.3, atol=1e-6)
    # Stored phi0 unchanged.
    assert weighter.phi0_current == 0.0


def test_get_weights_jax_matches() -> None:
    weighter = _weighter(normalize=False)
    np.testing.assert_allclose(np.asarray(weighter.get_weights_jax()), weighter.get_weights())


# ---------------------------------------------------------------------------
# update_phi0
# ---------------------------------------------------------------------------


def test_update_phi0_changes_weights() -> None:
    weighter = _weighter(normalize=False)
    params = np.zeros(11)  # n_per_angle=4, physical=7 -> phi0 at index 10
    params[10] = 90.0
    weighter.update_phi0(params, iteration=0)
    assert weighter.phi0_current == 90.0
    assert weighter.get_diagnostics()["update_count"] == 1


def test_update_phi0_skips_small_change() -> None:
    weighter = _weighter(normalize=False)
    params = np.zeros(11)
    params[10] = 0.05  # below the 0.1 deg threshold
    weighter.update_phi0(params, iteration=0)
    assert weighter.phi0_current == 0.0


def test_update_phi0_disabled_is_noop() -> None:
    weighter = _weighter(normalize=False, enable=False)
    params = np.zeros(11)
    params[10] = 90.0
    weighter.update_phi0(params, iteration=0)
    assert weighter.phi0_current == 0.0


def test_update_phi0_off_frequency_skips() -> None:
    weighter = _weighter(normalize=False, update_frequency=2)
    params = np.zeros(11)
    params[10] = 90.0
    weighter.update_phi0(params, iteration=1)  # 1 % 2 != 0 -> skipped
    assert weighter.phi0_current == 0.0


# ---------------------------------------------------------------------------
# weighted loss application
# ---------------------------------------------------------------------------


def test_apply_weights_to_loss_enabled() -> None:
    weighter = _weighter(normalize=False)
    residuals = jnp.ones(4)
    phi_indices = jnp.arange(4)
    # sum(w * 1) = 1.0 + 0.3 + 1.0 + 0.3 = 2.6
    loss = float(weighter.apply_weights_to_loss(residuals, phi_indices))
    assert loss == pytest.approx(2.6, abs=1e-5)


def test_apply_weights_to_loss_disabled() -> None:
    weighter = _weighter(normalize=False, enable=False)
    residuals = jnp.ones(4)
    phi_indices = jnp.arange(4)
    # disabled -> mean(r^2) * len = sum(r^2) = 4.0
    loss = float(weighter.apply_weights_to_loss(residuals, phi_indices))
    assert loss == pytest.approx(4.0, abs=1e-5)


def test_compute_weighted_mse_enabled() -> None:
    weighter = _weighter(normalize=False)
    residuals = jnp.ones(4)
    phi_indices = jnp.arange(4)
    # sum(w*r^2)/sum(w) = 2.6 / 2.6 = 1.0
    mse = float(weighter.compute_weighted_mse(residuals, phi_indices))
    assert mse == pytest.approx(1.0, abs=1e-5)


def test_compute_weighted_mse_disabled() -> None:
    weighter = _weighter(normalize=False, enable=False)
    residuals = jnp.array([1.0, 2.0, 3.0, 4.0])
    phi_indices = jnp.arange(4)
    mse = float(weighter.compute_weighted_mse(residuals, phi_indices))
    assert mse == pytest.approx(np.mean([1.0, 4.0, 9.0, 16.0]), abs=1e-5)


def test_get_diagnostics() -> None:
    diag = _weighter(normalize=False).get_diagnostics()
    assert diag["enabled"] is True
    assert diag["min_weight"] == 0.3
    assert diag["current_phi0"] == 0.0
    assert diag["weights_range"][0] == pytest.approx(0.3, abs=1e-6)
    assert diag["weights_range"][1] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# create_shear_weighting factory
# ---------------------------------------------------------------------------


def test_create_none_config_returns_none() -> None:
    assert sw.create_shear_weighting(np.array([0.0, 90.0]), 7, config=None) is None


def test_create_disabled_returns_none() -> None:
    out = sw.create_shear_weighting(
        np.array([0.0, 90.0]), 7, config={"shear_weighting": {"enable": False}}
    )
    assert out is None


def test_create_without_phi0_param_returns_none() -> None:
    out = sw.create_shear_weighting(
        np.array([0.0, 90.0]),
        n_physical=3,
        config={"shear_weighting": {"enable": True}},
        physical_param_names=["D0", "alpha", "D_offset"],  # no phi0
    )
    assert out is None


def test_create_resolves_phi0_index_from_names() -> None:
    names = ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]
    out = sw.create_shear_weighting(
        np.array([0.0, 90.0, 180.0]),
        n_physical=7,
        config={"shear_weighting": {"enable": True}},
        physical_param_names=names,
    )
    assert out is not None
    assert out.phi0_index == 6


def test_create_defaults_phi0_index_when_no_names() -> None:
    out = sw.create_shear_weighting(
        np.array([0.0, 90.0, 180.0]),
        n_physical=7,
        config={"shear_weighting": {"enable": True}},
    )
    assert out is not None
    assert out.phi0_index == 6
