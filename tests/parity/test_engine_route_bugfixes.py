"""Regression tests for engine-route ``two_component`` defects.

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

These tests are fixture-robust: the well-posed fixture used elsewhere has
UNIFORM true scaling, which hides Bug 1 numerically (the independent-DOF optimum
coincides with the uniform one), so Bug 1 is pinned by the optimizer DOF count
and wrapper usage, not by a chi-square delta on that fixture.
"""

from __future__ import annotations

import numpy as np

from tests.parity.test_engine_heterodyne_fit_parity import _make_well_posed_case
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
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
    fit_two_component_via_engine(
        model, c2, np.asarray(phi), _make_config("individual"), None
    )
    assert captured["x0_len"] == n_varying + 2 * n_phi
