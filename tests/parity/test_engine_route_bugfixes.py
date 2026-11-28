"""Regression tests for two engine-route ``two_component`` defects.

Both were surfaced by an adversarial review of the engine-route dispatch
(``xpcsjax/optimization/nlsq/heterodyne_engine_route.py``):

Bug 1 — averaged mode over-parameterized the scaling.
    For production ``auto`` at ``n_phi >= 3`` (engine ``averaged``) the
    contract is a COMPRESSED ``[physics | avg_contrast | avg_offset]`` problem —
    exactly **2 scaling DOF** shared across angles. The route instead broadcast
    the 2 averaged scalars to the engine's ``2*n_phi`` scaling-first layout and
    handed the optimizer ``n_varying + 2*n_phi`` DOF, letting it fit separate
    per-angle contrast/offset. It then compressed by keeping ONLY angle-0's
    fitted scalar and broadcasting it — discarding the other fitted values before
    scoring. The fix compresses the averaged scaling to the 2 shared scalars at
    the optimizer head (an inline broadcast in the engine route's ``residual_fn``)
    so the optimizer varies exactly 2 scaling DOF.

Bug 2 — single-angle 2D ``c2`` was accepted but mis-scored.
    The public dispatcher / old ``fit_nlsq_multi_phi`` accept a 2D single-angle
    ``c2`` matrix and normalize it to ``(1, N, N)``. The engine route passed the
    raw 2D array into ``_production_support_chi2`` →
    ``compute_multi_angle_residuals``, which vmaps over axis 0 — treating the
    leading TIME dimension as the angle batch (inconsistent vmap sizes vs the
    length-1 ``phi``/contrast/offset → raises, swallowed by the dispatcher's
    best-effort fallback). The fix normalizes ``c2`` (and 2D ``weights``) to
    ``(1, N, N)`` at the top of ``fit_two_component_via_engine``. For one angle the
    dispatcher also runs ``fit_nlsq_multi_phi`` and keeps the lower chi^2.

These tests are fixture-robust: the well-posed fixture used elsewhere has
UNIFORM true scaling, which hides Bug 1 numerically (the per-angle-DOF optimum
coincides with the uniform one), so Bug 1 is pinned by the optimizer DOF count
and wrapper usage, not by a chi-square delta on that fixture.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.parity.test_engine_heterodyne_fit_parity import _make_well_posed_case
from xpcsjax.config import ConfigManager
from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
from xpcsjax.optimization.nlsq import fit_nlsq, heterodyne_core, heterodyne_engine_route
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
# Bug 3 (fixed 2026-06-15) — max_nfev=None ("auto") must NOT bail the engine
# route. It mirrors the production joint paths (heterodyne_core.py:1232/:2684,
# heterodyne_constant_mode.py:244 — ``config.max_nfev * n_phi if
# config.max_nfev is not None else None``): None is passed through so nlsq
# auto-resolves 100*n_params for the engine's own joint solve. Previously the
# route RAISED NotImplementedError on None, silently routing every
# default-config two_component in-memory fit back to fit_nlsq_multi_phi and
# making the Task #16b procedural-parity route inert for the common case.
# ===========================================================================
def _spy_adapter_max_nfev(monkeypatch) -> dict:
    """Capture the ``max_nfev`` on the NLSQConfig handed to the joint solve."""
    from xpcsjax.optimization.nlsq import heterodyne_adapter

    captured: dict = {}
    real_fit = heterodyne_adapter.NLSQAdapter.fit

    def spy_fit(self, *args, **kwargs):
        cfg = kwargs.get("config")
        if cfg is None and len(args) >= 4:
            cfg = args[3]
        captured["max_nfev"] = cfg.max_nfev
        return real_fit(self, *args, **kwargs)

    monkeypatch.setattr(heterodyne_adapter.NLSQAdapter, "fit", spy_fit)
    return captured


def test_engine_route_runs_on_auto_max_nfev(monkeypatch):
    """``max_nfev=None`` (the NLSQConfig "auto" default) must NOT bail the engine
    route. It mirrors the production joint paths: None is passed through to the
    joint solve so nlsq auto-resolves 100*n_params. Previously the route raised
    NotImplementedError on None, silently routing every default-config
    two_component in-memory fit back to fit_nlsq_multi_phi (Task #16b inert).
    """
    model, c2, phi = _make_well_posed_case()
    cfg = _make_config("individual")
    cfg.max_nfev = None  # the NLSQConfig "auto" default

    captured = _spy_adapter_max_nfev(monkeypatch)
    result = fit_two_component_via_engine(model, c2, np.asarray(phi), cfg, None)

    assert result is not None
    assert captured["max_nfev"] is None, (
        "auto (None) max_nfev must pass through to the joint solve so nlsq "
        f"auto-resolves 100*n_params; got {captured['max_nfev']!r}"
    )


def test_engine_route_scales_explicit_max_nfev_by_n_phi(monkeypatch):
    """An explicit per-set ``max_nfev`` is scaled by ``n_phi`` for the combined
    joint solve, matching the production joint-path contract
    (NLSQConfig.max_nfev docstring)."""
    model, c2, phi = _make_well_posed_case()
    n_phi = len(phi)
    cfg = _make_config("individual")  # max_nfev=_PER_SET_NFEV

    captured = _spy_adapter_max_nfev(monkeypatch)
    fit_two_component_via_engine(model, c2, np.asarray(phi), cfg, None)

    assert captured["max_nfev"] == _PER_SET_NFEV * n_phi


# ===========================================================================
# Bug 1 — averaged mode must optimize a COMPRESSED 2-scaling-DOF problem
# ===========================================================================
def _spy_adapter_initial_params(monkeypatch) -> dict:
    """Patch ``NLSQAdapter.fit`` to record the optimizer-vector length / n_params
    actually handed to the solver, then delegate to the real fit."""
    from xpcsjax.optimization.nlsq import heterodyne_adapter

    captured: dict = {}
    real_fit = heterodyne_adapter.NLSQAdapter.fit

    def spy_fit(self, *args, **kwargs):
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
    """``averaged`` must hand the solver ``n_varying + 2`` DOF, NOT
    ``n_varying + 2*n_phi``. With the over-parameterized route the optimizer
    fit separate per-angle scaling — inconsistent with the averaged contract.
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


def test_averaged_mode_optimizes_compressed_two_scaling_dof(monkeypatch):
    """``averaged`` must optimize exactly ``n_varying + 2`` DOF (Bug 1).

    The original Task #14 fix routed averaged through a dedicated
    ``wrap_engine_averaged_residual``; that broadcast was later INLINED into the
    engine route's ``residual_fn`` (the wrapper module was retired). The behavior
    contract is unchanged and is what matters here: averaged compresses the
    ``2*n_phi`` engine scaling DOF down to the 2 shared scalars at the optimizer
    HEAD, so the solver varies ``n_varying + 2`` parameters (NOT ``2*n_phi`` and
    NOT a collapse to a single angle's fitted scalar).
    """
    model, c2, phi = _make_well_posed_case()
    n_varying = len(model.param_manager.varying_names)

    captured = _spy_adapter_initial_params(monkeypatch)
    fit_two_component_via_engine(model, c2, np.asarray(phi), _make_config("auto"), None)

    assert captured["x0_len"] == n_varying + 2, (
        "averaged engine route did not compress to 2 shared scaling DOF; the "
        "inline compressed-averaged broadcast must vary n_varying + 2 params (Bug 1)."
    )
    assert captured["n_params"] == n_varying + 2


def test_individual_mode_keeps_per_angle_scaling_dof(monkeypatch):
    """Guard: the fix must NOT change ``individual`` — it legitimately fits
    ``2*n_phi`` per-angle scaling DOF."""
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


def _single_angle_config() -> ConfigManager:
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
                "enable_cmaes": False,
                # the same solver settings as _make_config
                "method": "trf",
                "loss": "soft_l1",
                "ftol": 1e-8,
                "xtol": 1e-8,
                "gtol": 1e-8,
                "x_scale": "jac",
                "max_nfev": _PER_SET_NFEV,
            }
        },
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = Path(tmp_dir) / "fixture.yaml"
        import yaml

        cfg_path.write_text(yaml.safe_dump(cfg_dict))
        return ConfigManager(str(cfg_path))


def _build_single_angle_model() -> HeterodyneModel:
    cfg = _single_angle_config()
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
    c2_3d = _single_angle_c2(_build_single_angle_model(), float(phi[0]))

    res_3d = fit_two_component_via_engine(
        _build_single_angle_model(), c2_3d, phi, _make_config("auto"), None
    )
    res_2d = fit_two_component_via_engine(
        _build_single_angle_model(), c2_3d[0], phi, _make_config("auto"), None
    )

    assert np.isfinite(res_2d.chi_squared)
    assert np.asarray(res_2d.nlsq_diagnostics["chi2_per_angle"]).shape == (1,)
    assert np.isclose(res_2d.chi_squared, res_3d.chi_squared, rtol=1e-9, atol=1e-12), (
        f"2D c2 chi2 {res_2d.chi_squared!r} != 3D c2 chi2 {res_3d.chi_squared!r}; "
        "a 2D single-angle input must be normalized to (1, N, N) (Bug 2)."
    )
    np.testing.assert_allclose(res_2d.parameters, res_3d.parameters, rtol=1e-7, atol=1e-9)


def test_single_angle_2d_c2_no_worse_than_production(monkeypatch):
    """For one angle the dispatcher runs BOTH solvers and keeps the lower chi^2, so
    the fit is no worse than either, whichever of them a given CPU favours."""
    chi2: dict[str, float] = {}

    def record(module, name):
        real = getattr(module, name)

        def wrapper(*args, **kwargs):
            solved = real(*args, **kwargs)
            chi2[name] = solved.chi_squared
            return solved

        monkeypatch.setattr(module, name, wrapper)

    record(heterodyne_engine_route, "fit_two_component_via_engine")
    record(heterodyne_core, "fit_nlsq_multi_phi")
    phi = np.array([12.0], dtype=np.float64)
    c2_2d = _single_angle_c2(_build_single_angle_model(), float(phi[0]))[0]

    result = fit_nlsq({"c2": c2_2d, "phi": phi}, _single_angle_config())

    assert set(chi2) == {"fit_two_component_via_engine", "fit_nlsq_multi_phi"}
    assert result.chi_squared <= min(chi2.values()) * (1.0 + 1e-11), chi2


@pytest.mark.parametrize(
    ("engine_chi2", "production_chi2", "kept"),
    [
        (1.0, 0.5, "production"),
        (0.5, 1.0, "engine"),
        (1.0, 1.0, "production"),
        (float("nan"), 1.0, "production"),
        (1.0, float("nan"), "engine"),
        (1.0, None, "engine"),  # production raises
    ],
)
def test_single_angle_keeps_the_lower_chi2(monkeypatch, engine_chi2, production_chi2, kept):
    def stub(label, value):
        def solve(*args, **kwargs):
            if value is None:
                raise RuntimeError("solver failed")
            return SimpleNamespace(label=label, chi_squared=value, nlsq_diagnostics={})

        return solve

    monkeypatch.setattr(
        heterodyne_engine_route, "fit_two_component_via_engine", stub("engine", engine_chi2)
    )
    monkeypatch.setattr(heterodyne_core, "fit_nlsq_multi_phi", stub("production", production_chi2))
    phi = np.array([12.0], dtype=np.float64)
    c2 = _single_angle_c2(_build_single_angle_model(), float(phi[0]))

    assert fit_nlsq({"c2": c2, "phi": phi}, _single_angle_config()).label == kept


def test_multi_angle_fit_never_runs_the_second_solver(monkeypatch):
    calls: list[str] = []

    def stub(label):
        def solve(*args, **kwargs):
            calls.append(label)
            return SimpleNamespace(label=label, chi_squared=1.0, nlsq_diagnostics={})

        return solve

    monkeypatch.setattr(heterodyne_engine_route, "fit_two_component_via_engine", stub("engine"))
    monkeypatch.setattr(heterodyne_core, "fit_nlsq_multi_phi", stub("production"))
    phi = np.array([0.0, 45.0, 90.0], dtype=np.float64)
    c2 = np.concatenate([_single_angle_c2(_build_single_angle_model(), p) for p in phi])

    assert fit_nlsq({"c2": c2, "phi": phi}, _single_angle_config()).label == "engine"
    assert calls == ["engine"]


def test_single_angle_crashed_comparison_never_wins(monkeypatch):
    """A comparison whose solvers all crashed has no objective: keep the engine."""

    def crash(*_args, **_kwargs):
        raise RuntimeError("injected backend failure")

    monkeypatch.setattr(heterodyne_core.NLSQAdapter, "fit_jax", crash)
    monkeypatch.setattr(heterodyne_core.NLSQWrapper, "fit", crash)
    phi = np.array([12.0], dtype=np.float64)
    c2 = _single_angle_c2(_build_single_angle_model(), float(phi[0]))

    result = fit_nlsq({"c2": c2, "phi": phi}, _single_angle_config())

    assert result.device_info["adapter"] != "sequential_per_angle"
    assert np.isfinite(result.chi_squared) and result.chi_squared > 0.0


# ===========================================================================
# Bug (PR #27 review) — a fixed-but-NOT-tied physics parameter (fixed via
# active_parameters / set_vary) must also bail out of the engine route, not
# just a tied one. HeterodynePointEvaluator.eval_points passes the raw
# length-n_varying slice straight into compute_c2_heterodyne, which unpacks
# 14 positional params with no length check -- a short vector is silently
# clipped by the kernel (JAX statically clips an out-of-bounds index rather
# than raising). Tying already forces vary=False on the child, so the guard
# on ``len(varying_indices) != len(ALL_PARAM_NAMES)`` subsumes both cases.
# ===========================================================================
def test_engine_route_bails_on_fixed_untied_physics_param():
    """A plain fixed (non-tied) physics parameter must raise NotImplementedError
    from the engine route -- not silently run with a short/clipped vector --
    and the production fallback (``fit_nlsq_multi_phi``) must still handle it
    correctly (full 14-length physics, no shape mismatch)."""
    model, c2, phi = _make_well_posed_case()
    # Fix a physics parameter WITHOUT tying it to anything.
    model.param_manager.set_vary("D0_ref", False)
    assert not model.param_manager.tied_idx_pairs, "sanity: this config has no ties"
    assert len(model.param_manager.varying_indices) == 13

    cfg = _make_config("individual")

    import pytest

    with pytest.raises(NotImplementedError, match="requires all 14 physics parameters"):
        fit_two_component_via_engine(model, c2, phi, cfg, None)

    # The documented fallback target must still produce a well-formed result
    # for the identical fixed-untied config (full 14-physics parameter block).
    fallback = fit_nlsq_multi_phi(model, c2, list(phi), cfg, None)
    assert fallback.n_physics == 14
    assert np.isfinite(fallback.chi_squared)
