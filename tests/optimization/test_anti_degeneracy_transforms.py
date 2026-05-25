"""Round-trip coverage for anti-degeneracy parameter transforms (audit finding #4).

These transforms sit on the optimizer's per-iteration hot path (L1/L2 boundary)
yet had no numerical round-trip test — existing controller tests only assert layer
gating/activation. A sign or index error here silently corrupts every fit.
"""

from __future__ import annotations

import numpy as np

from xpcsjax.optimization.nlsq.anti_degeneracy_controller import (
    AntiDegeneracyController,
)

N_PHI = 7
N_PHYSICAL = 7
PHI = np.deg2rad(np.linspace(0.0, 180.0, N_PHI, endpoint=False))


def _build(per_angle_mode: str) -> AntiDegeneracyController:
    return AntiDegeneracyController.from_config(
        config_dict={"enable": True, "per_angle_mode": per_angle_mode},
        n_phi=N_PHI,
        phi_angles=PHI,
        n_physical=N_PHYSICAL,
        per_angle_scaling=True,
        is_laminar_flow=True,
        analysis_mode="laminar_flow",
    )


def _per_angle_params(contrast: np.ndarray, offset: np.ndarray) -> np.ndarray:
    physical = np.linspace(1.0, 7.0, N_PHYSICAL)
    return np.concatenate([contrast, offset, physical])


def test_fourier_round_trip_preserves_physical_block() -> None:
    ctrl = _build("fourier")
    contrast = 0.3 + 0.05 * np.cos(PHI)
    offset = 1.0 + 0.02 * np.sin(PHI)
    params = _per_angle_params(contrast, offset)

    fourier_params, _ = ctrl.transform_params_to_fourier(params)
    recovered = ctrl.transform_params_from_fourier(fourier_params)

    assert recovered.shape == params.shape
    assert np.all(np.isfinite(recovered))
    # Physical tail must pass through untouched regardless of Fourier truncation.
    assert np.allclose(recovered[-N_PHYSICAL:], params[-N_PHYSICAL:])


def test_constant_round_trip_is_exact_for_constant_scaling() -> None:
    ctrl = _build("constant")
    if not ctrl.use_constant:  # config did not enable constant mode on this build
        return
    contrast = np.full(N_PHI, 0.3)
    offset = np.full(N_PHI, 1.0)
    params = _per_angle_params(contrast, offset)

    constant_params = ctrl.transform_params_to_constant(params)
    # Collapsed layout: [contrast_mean, offset_mean, *physical]
    assert constant_params.shape[0] == 2 + N_PHYSICAL
    assert np.isclose(constant_params[0], 0.3)
    assert np.isclose(constant_params[1], 1.0)

    expanded = ctrl.transform_params_from_constant(constant_params)
    assert expanded.shape == params.shape
    assert np.allclose(expanded, params)  # constant input -> exact round-trip


def test_constant_collapse_uses_nanmean_and_preserves_physical() -> None:
    ctrl = _build("constant")
    if not ctrl.use_constant:
        return
    contrast = np.array([0.2, 0.4, np.nan, 0.4, 0.2, 0.4, 0.4])
    offset = np.full(N_PHI, 1.1)
    params = _per_angle_params(contrast, offset)

    collapsed = ctrl.transform_params_to_constant(params)
    assert np.isfinite(collapsed[0])  # NaN-safe mean
    assert np.allclose(collapsed[-N_PHYSICAL:], params[-N_PHYSICAL:])


def test_get_diagnostics_returns_mapping() -> None:
    ctrl = _build("fourier")
    diag = ctrl.get_diagnostics()
    assert isinstance(diag, dict)
