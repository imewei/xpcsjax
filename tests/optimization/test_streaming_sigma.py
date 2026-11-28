"""Fits that carry ``sigma`` must work on the streaming tier.

nlsq's ``AdaptiveHybridStreamingOptimizer`` rejects ``sigma``. These tests drive the
real optimizer, because a permissive double is what hid the failure.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

import xpcsjax
from tests.optimization import _heterodyne_fixtures as het
from tests.optimization import test_fixed_parameters_integration as laminar
from xpcsjax.config import ConfigManager
from xpcsjax.optimization.nlsq import fit_nlsq
from xpcsjax.optimization.nlsq import heterodyne_memory as het_memory
from xpcsjax.optimization.nlsq import wrapper as wrapper_module
from xpcsjax.optimization.nlsq.core import fit_nlsq_jax
from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
    build_heterodyne_stratified_data,
)
from xpcsjax.optimization.nlsq.strategies import heterodyne_hybrid_streaming as het_stream

_N_PHI, _N_T = 3, 10


def _spy_on_real_fit(monkeypatch) -> list[dict]:
    """Record what each real ``optimizer.fit`` call receives."""
    from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer

    calls: list[dict] = []
    original = AdaptiveHybridStreamingOptimizer.fit

    def fit(self, *args, **kwargs):
        x, y = kwargs["data_source"]
        calls.append({"sigma": kwargs.get("sigma"), "x": np.asarray(x), "y": np.asarray(y)})
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AdaptiveHybridStreamingOptimizer, "fit", fit)
    return calls


# ---------------------------------------------------------------------------
# laminar_flow
# ---------------------------------------------------------------------------


def _laminar_streaming_fit(monkeypatch, sigma: np.ndarray):
    data, config = laminar._hybrid_streaming_fake_stratify_and_config(
        monkeypatch=monkeypatch, n_phi=_N_PHI
    )
    stratify = wrapper_module.NLSQWrapper._apply_stratification_if_needed

    def with_sigma(self, *args, **kwargs):
        stratified = stratify(self, *args, **kwargs)
        stratified.sigma = sigma
        return stratified

    monkeypatch.setattr(wrapper_module.NLSQWrapper, "_apply_stratification_if_needed", with_sigma)
    return fit_nlsq_jax(data, config, use_adapter=False)


def test_laminar_sigma_fit_reports_its_stratified_fallback(monkeypatch):
    """The streaming solver refuses sigma; the stratified-LS result must say so."""
    calls = _spy_on_real_fit(monkeypatch)

    result = _laminar_streaming_fit(monkeypatch, np.full((_N_PHI, _N_T, _N_T), 0.01))

    assert len(calls) == 1 and calls[0]["sigma"] is not None
    assert result.recovery_actions == ["stratified_least_squares_method"]
    assert result.device_info["fallback_occurred"] is True
    assert "hybrid streaming: NotImplementedError" in result.device_info["fallback_reason"]
    assert "sigma" in result.device_info["fallback_reason"]
    assert result.convergence_status == "converged"
    assert np.isfinite(np.asarray(result.uncertainties)).all()


# ---------------------------------------------------------------------------
# two_component
# ---------------------------------------------------------------------------


def _heterodyne_streaming_fit(weights, n_t=8, mode="constant", masked_fill=None):
    model, c2, phi = het.make_synthetic_two_component(n_phi=_N_PHI, n_t=n_t)
    manager = model.param_manager
    if weights is not None:
        weights = weights(c2)
    if masked_fill is not None:
        c2 = np.where(weights > 0, c2, masked_fill)
    return het_stream.fit_with_stratified_hybrid_streaming_heterodyne(
        stratified_data=build_heterodyne_stratified_data(model, c2, phi, weights),
        model=model,
        physical_param_names=list(manager.varying_names),
        initial_params=manager.get_initial_values(),
        bounds=manager.get_bounds(),
        hybrid_config={"verbose": 0, "gauss_newton_max_iterations": 500},
        anti_degeneracy_config={"per_angle_mode": mode},
    )


def _masked(c2, value=1.0):
    weights = np.full(c2.shape, value)
    weights[:, 2, 5] = weights[:, 5, 2] = 0.0
    return weights


@pytest.mark.parametrize("value", [1.0, 4.0])
def test_heterodyne_masked_equal_weights_fit_the_observed_rows(monkeypatch, value):
    calls = _spy_on_real_fit(monkeypatch)

    # n_t=24: the 8-point fixture is too degenerate to converge, masked or not.
    popt, pcov, info = _heterodyne_streaming_fit(lambda c2: _masked(c2, value), n_t=24)

    assert calls[0]["sigma"] is None
    assert calls[0]["x"].shape[1] == 3
    assert len(calls[0]["y"]) == info["n_data_points"] == info["n_data_rows"] - 2 * _N_PHI
    assert info["success"]
    assert np.isfinite(popt).all()
    assert np.isnan(pcov).all() and info["covariance_is_placeholder"] is True


@pytest.mark.parametrize("mode", ["constant", "averaged"])
def test_heterodyne_masked_values_do_not_steer_the_fit(mode):
    """The quantile scaling is frozen or the start in these modes."""

    def long_lags_and_one_angle_masked(c2):
        lag = np.abs(np.subtract.outer(np.arange(c2.shape[1]), np.arange(c2.shape[1])))
        weights = np.where(lag >= 16, 0.0, 1.0) * np.ones_like(c2)
        weights[0] = 0.0
        return weights

    fits = [
        _heterodyne_streaming_fit(
            long_lags_and_one_angle_masked, n_t=24, mode=mode, masked_fill=fill
        )
        for fill in (np.nan, -100.0)
    ]

    np.testing.assert_allclose(fits[0][0], fits[1][0], rtol=1e-12)
    assert fits[0][2]["ssr"] == pytest.approx(fits[1][2]["ssr"], rel=1e-12)
    monitors = [fit[2]["anti_degeneracy"].get("gradient_monitor") for fit in fits]
    assert monitors[0] == monitors[1]


def test_heterodyne_equal_weights_match_the_unweighted_fit():
    unweighted, unweighted_cov, unweighted_info = _heterodyne_streaming_fit(None)
    weighted, weighted_cov, _ = _heterodyne_streaming_fit(lambda c2: np.full(c2.shape, 4.0))

    np.testing.assert_allclose(weighted, unweighted, rtol=1e-12)
    # Only the weighted fit is newly enabled; the unweighted report is unchanged.
    assert unweighted_info["covariance_is_placeholder"] is False
    assert np.isfinite(unweighted_cov).all()
    assert np.isnan(weighted_cov).all()


def test_heterodyne_uneven_weights_are_declined(monkeypatch):
    calls = _spy_on_real_fit(monkeypatch)

    def uneven(c2):
        weights = np.ones(c2.shape)
        weights[:, 2, 5] = weights[:, 5, 2] = 4.0
        return weights

    with pytest.raises(NotImplementedError, match="uneven sigma"):
        _heterodyne_streaming_fit(uneven)
    assert calls == []


def test_heterodyne_with_every_pixel_masked_is_a_data_error(monkeypatch):
    calls = _spy_on_real_fit(monkeypatch)

    with pytest.raises(ValueError, match="no observations"):
        _heterodyne_streaming_fit(np.zeros_like)
    assert calls == []


def test_heterodyne_fewer_observations_than_parameters():
    def four_pixels(c2):
        weights = np.zeros(c2.shape)
        weights[0, 1, 2:6] = 1.0
        return weights

    popt, pcov, info = _heterodyne_streaming_fit(four_pixels)

    assert info["n_data_points"] == 4 < len(popt)
    assert np.isnan(pcov).all() and info["covariance_is_placeholder"] is True


def test_heterodyne_public_fit_accepts_a_mask_on_the_streaming_tier(monkeypatch, tmp_path):
    select = het_memory.select_nlsq_strategy

    def streaming(*args, **kwargs):
        decision = select(*args, **kwargs)
        return type(decision)(
            **{**decision.__dict__, "strategy": het_memory.NLSQStrategy.STREAMING}
        )

    monkeypatch.setattr(het_memory, "select_nlsq_strategy", streaming)
    calls = _spy_on_real_fit(monkeypatch)
    template = yaml.safe_load(
        (Path(xpcsjax.__file__).parent / "config/templates/xpcsjax_two_component.yaml").read_text(
            encoding="utf-8"
        )
    )
    block = copy.deepcopy(template["optimization"]["nlsq"])
    block["analysis_mode"] = "two_component"
    block["hybrid_streaming"] = {
        **block.get("hybrid_streaming", {}),
        "enable": True,
        "verbose": 0,
        "enable_checkpoints": False,
    }
    config = het._config_dict(_N_PHI, 14)
    config["optimization"]["nlsq"] = block
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _, c2, phi = het.make_synthetic_two_component(n_phi=_N_PHI, n_t=14)

    result = fit_nlsq(
        {"c2_exp": c2, "phi_angles_list": phi, "weights": _masked(c2)}, ConfigManager(str(path))
    )

    assert len(calls) == 1
    assert np.isfinite(result.chi_squared)
    assert np.isnan(np.asarray(result.uncertainties)).all()
