"""Regression tests for two engine-route ``two_component`` defects.

Both were surfaced by an adversarial review of the engine-route dispatch
(``xpcsjax/optimization/nlsq/heterodyne_engine_route.py``):

Bug 1 — averaged mode over-parameterized the scaling.
    For production ``auto`` at ``n_phi >= 3`` (engine ``auto_averaged``) the
    contract is a COMPRESSED ``[physics | avg_contrast | avg_offset]`` problem —
    exactly **2 scaling DOF** shared across angles. The route instead broadcast
    the 2 averaged scalars to the engine's ``2*n_phi`` scaling-first layout and
    handed the optimizer ``n_varying + 2*n_phi`` DOF, letting it fit independent
    per-angle contrast/offset. It then compressed by keeping ONLY angle-0's
    fitted scalar and broadcasting it — discarding the other fitted values before
    scoring. The dedicated :func:`wrap_engine_averaged_residual` (Task #14)
    existed for exactly this but was never wired in. The fix routes
    ``auto_averaged`` through the wrapper so the optimizer varies 2 scaling DOF.

Bug 2 — single-angle 2D ``c2`` was accepted but mis-scored.
    The public dispatcher / old ``fit_nlsq_multi_phi`` accept a 2D single-angle
    ``c2`` matrix and normalize it to ``(1, N, N)``. The engine route passed the
    raw 2D array into ``_production_support_chi2`` →
    ``compute_multi_angle_residuals``, which vmaps over axis 0 — treating the
    leading TIME dimension as the angle batch (inconsistent vmap sizes vs the
    length-1 ``phi``/contrast/offset → raises, swallowed by the dispatcher's
    best-effort fallback). The fix normalizes ``c2`` (and 2D ``weights``) to
    ``(1, N, N)`` at the top of ``fit_two_component_via_engine``.

These tests are fixture-robust: the well-posed fixture used elsewhere has
UNIFORM true scaling, which hides Bug 1 numerically (the independent-DOF optimum
coincides with the uniform one), so Bug 1 is pinned by the optimizer DOF count
and wrapper usage, not by a chi-square delta on that fixture.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from tests.parity.test_engine_heterodyne_fit_parity import _make_well_posed_case
from xpcsjax.config import ConfigManager
from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi
from xpcsjax.optimization.nlsq.heterodyne_engine_route import (
    fit_two_component_via_engine,
)

_PER_SET_NFEV = 600


def _make_config(production_mode: str) -> NLSQConfig:
    cfg = NLSQConfig(
        method="trf",
        loss="soft_l1",
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        x_scale="jac",
        max_nfev=_PER_SET_NFEV,
        enable_cmaes=False,
        multistart=False,
    )
    cfg.per_angle_mode = production_mode
    return cfg


# ===========================================================================
# Bug 1 — averaged mode must optimize a COMPRESSED 2-scaling-DOF problem
# ===========================================================================
def _spy_adapter_initial_params(monkeypatch) -> dict:
    """Patch ``NLSQAdapter.fit`` to record the optimizer-vector length / n_params
    actually handed to the solver, then delegate to the real fit."""
    from xpcsjax.optimization.nlsq import heterodyne_adapter

    captured: dict = {}
    real_fit = heterodyne_adapter.NLSQAdapter.fit

    def spy_fit(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        ip = kwargs.get("initial_params")
        cfg = kwargs.get("config")
        if ip is None and len(args) >= 2:
            ip = args[1]
        if cfg is None and len(args) >= 4:
            cfg = args[3]
        captured["x0_len"] = int(np.asarray(ip).shape[0])
        captured["n_params"] = int(cfg.n_params)
        return real_fit(self, *args, **kwargs)

    monkeypatch.setattr(heterodyne_adapter.NLSQAdapter, "fit", spy_fit)
    return captured


def test_averaged_mode_optimizes_two_compressed_scaling_dof(monkeypatch):
    """``auto_averaged`` must hand the solver ``n_varying + 2`` DOF, NOT
    ``n_varying + 2*n_phi``. With the over-parameterized route the optimizer
    fit independent per-angle scaling — inconsistent with the averaged contract.
    """
    model, c2, phi = _make_well_posed_case()
    n_phi = len(phi)
    n_varying = len(model.param_manager.varying_names)
    assert n_phi >= 3, "fixture must resolve auto -> averaged"

    captured = _spy_adapter_initial_params(monkeypatch)
    fit_two_component_via_engine(model, c2, np.asarray(phi), _make_config("auto"), None)

    assert captured["x0_len"] == n_varying + 2, (
        f"averaged engine route handed the optimizer {captured['x0_len']} DOF; "
        f"expected the COMPRESSED {n_varying + 2} (= n_varying + 2). A value of "
        f"{n_varying + 2 * n_phi} (= n_varying + 2*n_phi) means the route is "
        "over-parameterizing averaged scaling (Bug 1)."
    )
    assert captured["n_params"] == n_varying + 2


def test_averaged_mode_uses_compressed_wrapper(monkeypatch):
    """``auto_averaged`` must route through ``wrap_engine_averaged_residual``
    (Task #14), and the wrapped residual must receive compressed
    ``n_varying + 2`` vectors."""
    from xpcsjax.optimization.nlsq import heterodyne_averaged_wrapper

    model, c2, phi = _make_well_posed_case()
    n_varying = len(model.param_manager.varying_names)
    calls: dict = {"count": 0, "residual_input_len": None}

    real_wrap = heterodyne_averaged_wrapper.wrap_engine_averaged_residual

    def spy_wrap(engine, *, n_physics, n_phi):  # noqa: ANN001
        calls["count"] += 1
        inner = real_wrap(engine, n_physics=n_physics, n_phi=n_phi)

        def recording(x):  # noqa: ANN001
            if calls["residual_input_len"] is None:
                calls["residual_input_len"] = int(np.asarray(x).shape[0])
            return inner(x)

        return recording

    monkeypatch.setattr(
        "xpcsjax.optimization.nlsq.heterodyne_engine_route.wrap_engine_averaged_residual",
        spy_wrap,
        raising=False,
    )

    fit_two_component_via_engine(model, c2, np.asarray(phi), _make_config("auto"), None)

    assert calls["count"] >= 1, (
        "averaged engine route did not call wrap_engine_averaged_residual; the "
        "compressed-averaged wrapper (Task #14) must be wired in (Bug 1)."
    )
    assert calls["residual_input_len"] == n_varying + 2


def test_individual_mode_keeps_per_angle_scaling_dof(monkeypatch):
    """Guard: the fix must NOT change ``individual`` — it legitimately fits
    ``2*n_phi`` independent scaling DOF."""
    model, c2, phi = _make_well_posed_case()
    n_phi = len(phi)
    n_varying = len(model.param_manager.varying_names)

    captured = _spy_adapter_initial_params(monkeypatch)
    fit_two_component_via_engine(model, c2, np.asarray(phi), _make_config("individual"), None)
    assert captured["x0_len"] == n_varying + 2 * n_phi


# ===========================================================================
# Bug 2 — single-angle 2D c2 must be accepted and scored as (1, N, N)
# ===========================================================================
_SA_N_T = 12
_SA_Q = 0.0054
_SA_DT = 1.0
# beta (v_beta) defaults to 1.0 with bounds [0, 2]. The single-angle problem has
# far less data than the multi-angle fixture, so its noiseless basin around the
# x0 default is narrower: only truths within ~0.05 of 1.0 are recovered by the
# plain local solve. Use a -0.05 shift (vs the multi-angle fixture's -0.10).
_SA_TRUE_PERTURB = {
    "D0_ref": 1.10e4,
    "alpha_ref": 0.10,
    "D0_sample": 1.05e4,
    "beta": 0.95,
    "f0": 0.55,
}


def _build_single_angle_model() -> HeterodyneModel:
    cfg_dict = {
        "analysis_mode": "two_component",
        "analyzer_parameters": {
            "dt": _SA_DT,
            "start_frame": 1,
            "end_frame": _SA_N_T,
            "scattering": {"wavevector_q": _SA_Q},
        },
        "scaling": {
            "n_angles": 1,
            "mode": "constant",
            "initial_contrast": 0.3,
            "initial_offset": 1.0,
        },
        "optimization": {
            "nlsq": {
                "analysis_mode": "two_component",
                "max_iterations": 30,
                "enable_cmaes": False,
            }
        },
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = Path(tmp_dir) / "fixture.yaml"
        import yaml

        cfg_path.write_text(yaml.safe_dump(cfg_dict))
        cfg = ConfigManager(str(cfg_path))
    assert cfg.config is not None
    return HeterodyneModel.from_config(cfg.config)


def _single_angle_c2(model: HeterodyneModel, phi_val: float) -> np.ndarray:
    """Noiseless single-angle correlation at a perturbed truth, uniform scaling.
    Returns shape ``(1, N, N)``."""
    pm = model.param_manager
    names = list(pm.varying_names)
    true_full = np.asarray(pm.get_full_values(), dtype=np.float64).copy()
    for name, val in _SA_TRUE_PERTURB.items():
        if name in names:
            true_full[names.index(name)] = val
    c2 = np.asarray(
        model.compute_correlation(
            phi_angle=float(phi_val),
            params=true_full,
            contrast=0.30,
            offset=1.00,
            angle_idx=0,
        ),
        dtype=np.float64,
    )
    return c2[np.newaxis, ...]


def test_single_angle_2d_c2_equivalent_to_3d():
    """A 2D ``(N, N)`` single-angle ``c2`` must produce the SAME result as the
    explicit ``(1, N, N)`` stack — not raise and not silently mis-score."""
    phi = np.array([12.0], dtype=np.float64)
    # Generate data once (deterministic), fit with fresh models (state mutates).
    c2_3d = _single_angle_c2(_build_single_angle_model(), float(phi[0]))
    c2_2d = c2_3d[0]

    eng_3d = fit_two_component_via_engine(
        _build_single_angle_model(), c2_3d, phi, _make_config("auto"), None
    )
    eng_2d = fit_two_component_via_engine(
        _build_single_angle_model(), c2_2d, phi, _make_config("auto"), None
    )

    assert np.isfinite(eng_2d.chi_squared)
    assert np.asarray(eng_2d.nlsq_diagnostics["chi2_per_angle"]).shape == (1,)
    assert np.isclose(eng_2d.chi_squared, eng_3d.chi_squared, rtol=1e-9, atol=1e-12), (
        f"2D c2 chi2 {eng_2d.chi_squared!r} != 3D c2 chi2 {eng_3d.chi_squared!r}; "
        "a 2D single-angle input must be normalized to (1, N, N) (Bug 2)."
    )
    np.testing.assert_allclose(eng_2d.parameters, eng_3d.parameters, rtol=1e-7, atol=1e-9)


def test_single_angle_2d_c2_no_worse_than_production():
    """2D single-angle parity against the old ``fit_nlsq_multi_phi`` path."""
    phi = np.array([12.0], dtype=np.float64)
    c2_2d = _single_angle_c2(_build_single_angle_model(), float(phi[0]))[0]

    eng = fit_two_component_via_engine(
        _build_single_angle_model(), c2_2d, phi, _make_config("auto"), None
    )
    ref = fit_nlsq_multi_phi(
        _build_single_angle_model(), c2_2d, list(phi), _make_config("auto"), None
    )
    assert np.isfinite(eng.chi_squared) and np.isfinite(ref.chi_squared)
    # Both reach the noiseless single-angle minimum (~machine zero), so the
    # no-worse contract needs an absolute floor — a relative-only check is
    # meaningless when production lands at ~1e-19. The atol still catches a
    # genuinely worse engine (e.g. trapped at SSR ~1e-1).
    assert eng.chi_squared <= ref.chi_squared * (1.0 + 1e-3) + 1e-9, (
        f"engine 2D chi2 {eng.chi_squared!r} strictly worse than production {ref.chi_squared!r}"
    )
