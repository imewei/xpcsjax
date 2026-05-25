"""Tests for three NLSQ support modules.

* parameter_index_mapper: parameter-group index bookkeeping across the three
  per-angle modes (constant / fourier / individual), with mutual-exclusion and
  bounds validation.
* jacobian: jacfwd-based Jacobian statistics, condition number, sensitivity,
  and gradient-noise estimation (well- vs ill-conditioned QR branch, failure
  fallbacks).
* memory: system-memory detection, adaptive threshold (argument/env/clamp/
  fallback sources), and the memory-based strategy decision tree.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

from xpcsjax.optimization.nlsq import jacobian as jac
from xpcsjax.optimization.nlsq import memory as mem
from xpcsjax.optimization.nlsq.parameter_index_mapper import ParameterIndexMapper

# ===========================================================================
# ParameterIndexMapper
# ===========================================================================


def _fourier_stub(n_coeffs_per_param: int = 5) -> Any:
    return cast(
        Any,
        SimpleNamespace(
            use_fourier=True,
            n_coeffs_per_param=n_coeffs_per_param,
            n_coeffs=2 * n_coeffs_per_param,
        ),
    )


def test_mapper_individual_mode() -> None:
    m = ParameterIndexMapper(n_phi=23, n_physical=7)
    assert m.mode_name == "individual"
    assert m.use_fourier is False
    assert m.n_per_angle_total == 46
    assert m.n_per_group == 23
    assert m.get_group_indices() == [(0, 23), (23, 46)]
    assert m.get_physical_indices() == list(range(46, 53))
    assert m.total_params == 53


def test_mapper_constant_mode() -> None:
    m = ParameterIndexMapper(n_phi=23, n_physical=7, use_constant=True)
    assert m.mode_name == "constant"
    assert m.n_per_group == 1
    assert m.n_per_angle_total == 2
    assert m.get_group_indices() == [(0, 1), (1, 2)]
    assert m.total_params == 9


def test_mapper_fourier_mode() -> None:
    m = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=_fourier_stub(5))
    assert m.mode_name == "fourier"
    assert m.use_fourier is True
    assert m.n_per_group == 5
    assert m.n_per_angle_total == 10
    assert m.get_group_indices() == [(0, 5), (5, 10)]


def test_mapper_validation_errors() -> None:
    with pytest.raises(ValueError, match="n_phi must be >= 1"):
        ParameterIndexMapper(n_phi=0, n_physical=7)
    with pytest.raises(ValueError, match="n_physical must be >= 1"):
        ParameterIndexMapper(n_phi=5, n_physical=0)
    with pytest.raises(ValueError, match="Cannot use both"):
        ParameterIndexMapper(n_phi=5, n_physical=7, fourier=_fourier_stub(), use_constant=True)


def test_mapper_per_angle_indices_and_slices() -> None:
    m = ParameterIndexMapper(n_phi=3, n_physical=2)  # 6 per-angle + 2 physical
    assert m.get_per_angle_indices() == list(range(6))
    per_angle_slice, physical_slice = m.get_covariance_slice_indices()
    assert per_angle_slice == slice(0, 6)
    assert physical_slice == slice(6, 8)


def test_mapper_validate_indices() -> None:
    m = ParameterIndexMapper(n_phi=3, n_physical=2)  # total 8
    assert m.validate_indices(np.zeros(8)) is True
    with pytest.raises(ValueError, match="exceeds parameter count"):
        m.validate_indices(np.zeros(4))  # groups need indices up to 6


def test_mapper_diagnostics() -> None:
    diag = ParameterIndexMapper(n_phi=3, n_physical=2).get_diagnostics()
    assert diag["mode_name"] == "individual"
    assert diag["total_params"] == 8
    assert diag["group_indices"] == [(0, 3), (3, 6)]


# ===========================================================================
# jacobian
# ===========================================================================


def _linear_residual(x: np.ndarray, a: float, b: float) -> jnp.ndarray:
    return jnp.asarray(a) * jnp.asarray(x) + jnp.asarray(b)


def test_jacobian_stats_well_conditioned() -> None:
    jtj, col_norms = jac.compute_jacobian_stats(
        _linear_residual, np.array([1.0, 2.0, 3.0]), np.array([2.0, 1.0]), 1.0
    )
    assert jtj is not None and col_norms is not None
    assert jtj.shape == (2, 2)
    np.testing.assert_allclose(jtj, jtj.T, atol=1e-10)  # symmetric


def test_jacobian_stats_ill_conditioned_qr_branch() -> None:
    # Second column ~1e-8 -> condition number > 1e6 -> QR path.
    def resid(x: np.ndarray, a: float, b: float) -> jnp.ndarray:
        return jnp.asarray(a) * jnp.asarray(x) + jnp.asarray(b) * 1e-8

    jtj, col_norms = jac.compute_jacobian_stats(
        resid, np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0]), 1.0
    )
    assert jtj is not None  # QR branch still returns a valid J^T J
    assert jtj.shape == (2, 2)


def test_jacobian_stats_jax_residual_attr() -> None:
    class _R:
        def jax_residual(self, p: jnp.ndarray) -> jnp.ndarray:
            return p[0] * jnp.ones(3) + p[1]

    jtj, col_norms = jac.compute_jacobian_stats(
        cast(Any, _R()), np.array([0.0]), np.array([2.0, 1.0]), 1.0
    )
    assert jtj is not None and jtj.shape == (2, 2)


def test_jacobian_stats_failure_returns_none() -> None:
    def boom(x: np.ndarray, *p: float) -> jnp.ndarray:
        raise ValueError("residual failure")

    assert jac.compute_jacobian_stats(boom, np.array([1.0]), np.array([1.0]), 1.0) == (
        None,
        None,
    )


def test_jacobian_condition_number() -> None:
    cond = jac.compute_jacobian_condition_number(
        _linear_residual, np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0])
    )
    assert cond is not None and cond >= 1.0


def test_jacobian_condition_number_failure() -> None:
    def boom(x: np.ndarray, *p: float) -> jnp.ndarray:
        raise RuntimeError("fail")

    assert jac.compute_jacobian_condition_number(boom, np.array([1.0]), np.array([1.0])) is None


def test_analyze_parameter_sensitivity() -> None:
    sens = jac.analyze_parameter_sensitivity(
        _linear_residual, np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0]), ["a", "b"]
    )
    assert set(sens) == {"a", "b"}
    assert all(0.0 <= v <= 1.0 for v in sens.values())
    assert max(sens.values()) == pytest.approx(1.0)  # normalized to max


def test_analyze_parameter_sensitivity_zero_gradient() -> None:
    # Residual independent of params -> all column norms zero -> zeros.
    def constant_resid(x: np.ndarray, a: float, b: float) -> jnp.ndarray:
        return jnp.asarray(x)

    sens = jac.analyze_parameter_sensitivity(
        constant_resid, np.array([1.0, 2.0]), np.array([1.0, 1.0]), ["a", "b"]
    )
    assert sens == {"a": 0.0, "b": 0.0}


def test_analyze_parameter_sensitivity_failure_returns_empty() -> None:
    def boom(x: np.ndarray, *p: float) -> jnp.ndarray:
        raise ValueError("fail")

    assert jac.analyze_parameter_sensitivity(boom, np.array([1.0]), np.array([1.0]), ["a"]) == {}


def test_estimate_gradient_noise() -> None:
    noise = jac.estimate_gradient_noise(
        _linear_residual, np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0]), n_samples=3
    )
    assert noise is not None and noise >= 0.0


def test_estimate_gradient_noise_failure() -> None:
    def boom(x: np.ndarray, *p: float) -> jnp.ndarray:
        raise ValueError("fail")

    assert jac.estimate_gradient_noise(boom, np.array([1.0]), np.array([1.0])) is None


# ===========================================================================
# memory
# ===========================================================================


def test_detect_total_system_memory() -> None:
    total = mem.detect_total_system_memory()
    assert total is not None and total > 0  # psutil available in test env


def test_estimate_peak_memory_gb_formula() -> None:
    peak = mem.estimate_peak_memory_gb(1_000_000, 53)
    expected = 1_000_000 * 53 * 8 * 6.5 / (1024**3)
    assert peak == pytest.approx(expected)


def test_adaptive_threshold_argument_source() -> None:
    threshold, info = mem.get_adaptive_memory_threshold(memory_fraction=0.5)
    assert info["source"] == "argument"
    assert info["memory_fraction"] == 0.5
    assert threshold == pytest.approx(info["total_memory_gb"] * 0.5)


def test_adaptive_threshold_env_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(mem.MEMORY_FRACTION_ENV_VAR, "0.6")
    _, info = mem.get_adaptive_memory_threshold()
    assert info["source"] == "env"
    assert info["memory_fraction"] == pytest.approx(0.6)


def test_adaptive_threshold_default_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(mem.MEMORY_FRACTION_ENV_VAR, raising=False)
    _, info = mem.get_adaptive_memory_threshold()
    assert info["source"] == "default"
    assert info["memory_fraction"] == mem.DEFAULT_MEMORY_FRACTION


def test_adaptive_threshold_clamps_fraction() -> None:
    with pytest.warns(UserWarning, match="clamped"):
        _, info = mem.get_adaptive_memory_threshold(memory_fraction=0.99)
    assert info["memory_fraction"] == mem.MAX_MEMORY_FRACTION


def test_adaptive_threshold_invalid_env_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(mem.MEMORY_FRACTION_ENV_VAR, "not-a-float")
    with pytest.warns(UserWarning, match="Invalid"):
        _, info = mem.get_adaptive_memory_threshold()
    assert info["memory_fraction"] == mem.DEFAULT_MEMORY_FRACTION


def test_adaptive_threshold_fallback_on_detection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mem, "detect_total_system_memory", lambda: None)
    with pytest.warns(UserWarning, match="Could not detect"):
        threshold, info = mem.get_adaptive_memory_threshold(memory_fraction=0.5)
    assert threshold == mem.FALLBACK_THRESHOLD_GB
    assert info["detection_method"] == "fallback"


def _patch_threshold(monkeypatch: pytest.MonkeyPatch, gb: float) -> None:
    monkeypatch.setattr(mem, "get_adaptive_memory_threshold", lambda *a, **k: (gb, {}))


def test_select_strategy_standard(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_threshold(monkeypatch, 1.0)
    decision = mem.select_nlsq_strategy(1000, 10)
    assert decision.strategy is mem.NLSQStrategy.STANDARD
    assert "fits" in decision.reason


def test_select_strategy_out_of_core(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_threshold(monkeypatch, 1.0)
    # 10M points x 53 params -> peak >> 1 GB but index (0.08 GB) < 1 GB.
    decision = mem.select_nlsq_strategy(10_000_000, 53)
    assert decision.strategy is mem.NLSQStrategy.OUT_OF_CORE


def test_select_strategy_hybrid_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_threshold(monkeypatch, 1.0)
    # 200M points -> index array (1.49 GB) > 1 GB threshold.
    decision = mem.select_nlsq_strategy(200_000_000, 53)
    assert decision.strategy is mem.NLSQStrategy.HYBRID_STREAMING


def test_select_strategy_zero_params_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_threshold(monkeypatch, 1.0)
    decision = mem.select_nlsq_strategy(1000, 0)
    assert decision.peak_memory_gb == 0.0
    assert decision.strategy is mem.NLSQStrategy.STANDARD
