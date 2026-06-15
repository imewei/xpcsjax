"""Regression: static modes are pinned to the ``individual`` per-angle layout
on every fit path.

Static modes (``static_isotropic`` / ``static_anisotropic``) have no flow
direction, so the laminar ``auto -> averaged`` scaling compression is a no-op
for them. The blessed invariant is "static keeps individual" — the optimized
vector is ``n_physics + 2*n_phi`` regardless of the requested per-angle mode.
"Static unification" (letting static honor auto/averaged/constant) is DEFERRED
(spec §9); see the skips in ``tests/parity/test_phase5_default_no_worse.py`` and
``tests/characterization/test_homodyne_equivalence.py``.

These tests guard the four fit paths against drift:
  - standard NLSQ (in-memory): wrapper Step 6.5 hard-wires static -> individual
  - CMA-ES / stratified-LS: the AntiDegeneracyController is gated to
    laminar_flow / two_component, so static never compresses scaling
  - hybrid streaming: ``_resolve_streaming_per_angle_mode`` pins static ->
    individual (this is the bug that previously let a streamed static fit
    silently resolve auto -> averaged, making DOF depend on dataset size)
and against the ``static_anisotropic`` template drifting back to ``auto``.
"""
from __future__ import annotations

import inspect
import pathlib

import numpy as np
import pytest
import yaml

_STATIC_PHYS = 3  # static_isotropic / static_anisotropic: [D0, alpha, D_offset]
_TEMPLATE = pathlib.Path(
    "xpcsjax/config/templates/xpcsjax_static_anisotropic.yaml"
)


# ---------------------------------------------------------------------------
# Hybrid-streaming path — the fix. ``_resolve_streaming_per_angle_mode`` pins
# static to individual; laminar still honors the full resolver.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("requested", ["auto", "averaged", "constant", "individual"])
@pytest.mark.parametrize("n_phi", [2, 3, 23])
def test_streaming_static_always_individual(requested, n_phi):
    from xpcsjax.optimization.nlsq.strategies.hybrid_streaming import (
        _resolve_streaming_per_angle_mode,
    )

    assert (
        _resolve_streaming_per_angle_mode(
            requested, n_phi, 3, is_laminar_flow=False
        )
        == "individual"
    )


def test_streaming_laminar_still_honors_resolver():
    from xpcsjax.optimization.nlsq.strategies.hybrid_streaming import (
        _resolve_streaming_per_angle_mode,
    )

    assert _resolve_streaming_per_angle_mode("auto", 3, 3, is_laminar_flow=True) == "averaged"
    assert _resolve_streaming_per_angle_mode("auto", 2, 3, is_laminar_flow=True) == "individual"
    assert _resolve_streaming_per_angle_mode("constant", 5, 3, is_laminar_flow=True) == "constant"
    assert _resolve_streaming_per_angle_mode("averaged", 5, 3, is_laminar_flow=True) == "averaged"


def test_streaming_function_routes_through_static_pin_helper():
    """Pin the wiring: the streaming fit must resolve its mode via the helper,
    never re-inline ``resolve_per_angle_mode`` (which would skip the static pin)."""
    from xpcsjax.optimization.nlsq.strategies import hybrid_streaming

    src = inspect.getsource(hybrid_streaming.fit_with_stratified_hybrid_streaming)
    assert "_resolve_streaming_per_angle_mode(" in src
    # The bare resolver must NOT be called directly in the fit body (it would
    # bypass the static pin); it is reachable only inside the helper.
    assert "resolve_per_angle_mode(" not in src.replace(
        "_resolve_streaming_per_angle_mode(", ""
    )


# ---------------------------------------------------------------------------
# CMA-ES + stratified-LS paths — both gate the AntiDegeneracyController to
# laminar_flow / two_component. For static the controller never enables, so the
# constant/averaged compression branches are unreachable -> individual.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["static_anisotropic", "static_isotropic"])
def test_controller_disabled_for_static_so_no_compression(mode):
    from xpcsjax.optimization.nlsq.anti_degeneracy_controller import (
        AntiDegeneracyController,
    )

    phi = np.deg2rad(np.linspace(0.0, 120.0, 3, endpoint=False))
    ctrl = AntiDegeneracyController.from_config(
        config_dict={
            "enable": True,
            "per_angle_mode": "auto",  # would resolve averaged @ n_phi=3 if honored
            "constant_scaling_threshold": 3,
        },
        n_phi=3,
        phi_angles=phi,
        n_physical=_STATIC_PHYS,
        per_angle_scaling=True,
        is_laminar_flow=False,
        analysis_mode=mode,
    )
    # Controller never initializes for static -> the CMA-ES guard
    # (`is_enabled and use_constant`) and the stratified-LS gate both fall
    # through to the dense individual layout.
    assert ctrl.is_enabled is False


# ---------------------------------------------------------------------------
# Standard in-memory path — small static_anisotropic fit must produce the dense
# individual vector (n_physics + 2*n_phi) even though the config requests auto.
# ---------------------------------------------------------------------------
def _static_cfg(per_angle_mode: str, n_t: int = 8):
    from xpcsjax.config import ConfigManager

    ad = {
        "enable": True,
        "per_angle_mode": per_angle_mode,
        "constant_scaling_threshold": 3,
    }
    cfg = {
        "analysis_mode": "static_anisotropic",
        "analyzer_parameters": {
            "dt": 0.1, "start_frame": 1, "end_frame": n_t,
            "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": n_t},
            "scattering": {"wavevector_q": 0.0237},
            "geometry": {"stator_rotor_gap": 2000000},
        },
        "initial_parameters": {
            "parameter_names": ["D0", "alpha", "D_offset"],
            "values": [1000.0, 0.5, 10.0],
        },
        "optimization": {
            "method": "nlsq",
            "nlsq": {
                "analysis_mode": "static_anisotropic", "max_iterations": 30,
                "loss": "linear",
                "cmaes": {"enable": False, "auto_select": False},
                "multi_start": {"enable": False},
                "anti_degeneracy": ad,
            },
            "stratification": {"enabled": False},
        },
    }
    return ConfigManager(config_override=cfg)


def test_standard_inmemory_static_is_individual_despite_auto():
    from xpcsjax.core.homodyne_model import HomodyneModel
    from xpcsjax.optimization.nlsq import fit_nlsq

    n_phi, n_t = 4, 8
    phi = np.linspace(0.0, 90.0, n_phi, dtype=np.float64)
    t = np.linspace(0.0, float(n_t - 1), n_t, dtype=np.float64)
    true = np.array([1000.0, 0.5, 10.0], dtype=np.float64)
    cfg = _static_cfg("auto", n_t)  # auto would be averaged @ n_phi>=3 IF honored
    model = HomodyneModel(cfg.config)
    c2 = np.asarray(model.compute_c2(true, phi, contrast=0.3, offset=1.0), dtype=np.float64)
    c2 = c2 + np.random.default_rng(5).normal(0.0, 5e-4, size=c2.shape)
    data = {"phi_angles_list": phi, "c2_exp": c2, "t1": t, "t2": t,
            "wavevector_q_list": np.array([0.0237], dtype=np.float64)}

    res = fit_nlsq(data, cfg)
    params = np.asarray(res.parameters, dtype=np.float64)
    # Individual (dense) layout, NOT averaged (which would be _STATIC_PHYS + 2).
    assert params.shape[0] == 2 * n_phi + _STATIC_PHYS
    assert np.all(np.isfinite(params))
    diag = dict(res.nlsq_diagnostics or {})
    if "per_angle_mode" in diag:
        assert diag["per_angle_mode"] == "individual"
    if diag.get("n_optimized") is not None:
        assert int(diag["n_optimized"]) == 2 * n_phi


# ---------------------------------------------------------------------------
# Template — static_anisotropic config must ship the pinned mode so config
# intent matches enforced behavior on every path.
# ---------------------------------------------------------------------------
def test_static_anisotropic_template_pins_individual():
    cfg = yaml.safe_load(_TEMPLATE.read_text(encoding="utf-8"))
    ad = cfg["optimization"]["nlsq"]["anti_degeneracy"]
    assert ad["per_angle_mode"] == "individual", (
        "static_anisotropic must pin per_angle_mode to 'individual' until static "
        "unification (spec §9) lands; 'auto' silently resolves averaged on the "
        "streaming path."
    )


# ---------------------------------------------------------------------------
# Shared static-pin resolver (single source of truth for the invariant). Both
# the streaming fit and the large-data reduced-chi2 DOF computations resolve
# through it so a static fit's optimized param count AND its DOF/reduced-chi2
# are individual everywhere, regardless of dataset size or requested mode.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("requested", ["auto", "averaged", "constant", "individual"])
@pytest.mark.parametrize("n_phi", [2, 3, 23])
def test_resolve_static_pinned_forces_individual(requested, n_phi):
    from xpcsjax.optimization.nlsq.per_angle_mode import (
        resolve_per_angle_mode_static_pinned,
    )

    assert (
        resolve_per_angle_mode_static_pinned(requested, n_phi, 3, is_laminar_flow=False)
        == "individual"
    )


def test_resolve_static_pinned_laminar_honors_resolver():
    from xpcsjax.optimization.nlsq.per_angle_mode import (
        resolve_per_angle_mode_static_pinned as _r,
    )

    assert _r("auto", 3, 3, is_laminar_flow=True) == "averaged"
    assert _r("auto", 2, 3, is_laminar_flow=True) == "individual"
    assert _r("constant", 5, 3, is_laminar_flow=True) == "constant"
    assert _r("averaged", 5, 3, is_laminar_flow=True) == "averaged"


# ---------------------------------------------------------------------------
# Reduced-chi2 DOF — the bug. The large-data DOF computations resolved the RAW
# config token (static_isotropic ships "constant"), giving DOF = n_physical (3)
# while the optimizer actually fit the dense individual vector (2*n_phi + 3).
# The static pin makes the DOF match the fit.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("token", ["constant", "averaged", "auto"])
def test_static_dof_uses_individual_param_count(token):
    from xpcsjax.optimization.nlsq.per_angle_mode import (
        effective_constrained_dof,
        resolve_per_angle_mode_static_pinned,
    )

    n_phi, n_physical = 6, 3
    resolved = resolve_per_angle_mode_static_pinned(token, n_phi, 1, is_laminar_flow=False)
    assert resolved == "individual"
    # individual -> None so the caller falls back to len(popt) = 2*n_phi + n_physical,
    # NOT the n_physical the inert "constant" token would have produced.
    assert effective_constrained_dof(resolved, n_phi=n_phi, n_physical=n_physical) is None


def test_wrapper_dof_sites_resolve_through_static_pin():
    """Wiring: every large-data reduced-chi2 DOF computation in the wrapper must
    resolve its per-angle mode through the static-pinned resolver, never the bare
    resolver (which skips the pin and mis-sizes static DOF)."""
    src = pathlib.Path("xpcsjax/optimization/nlsq/wrapper.py").read_text(encoding="utf-8")
    assert src.count("effective_constrained_dof") >= 3
    assert "resolve_per_angle_mode_static_pinned" in src


# ---------------------------------------------------------------------------
# Template — static_isotropic must ALSO pin individual (it shipped "constant",
# an inert token on the optimizer path that mis-sized reduced-chi2 DOF).
# ---------------------------------------------------------------------------
def test_static_isotropic_template_pins_individual():
    iso = pathlib.Path("xpcsjax/config/templates/xpcsjax_static_isotropic.yaml")
    cfg = yaml.safe_load(iso.read_text(encoding="utf-8"))
    ad = cfg["optimization"]["nlsq"]["anti_degeneracy"]
    assert ad["per_angle_mode"] == "individual", (
        "static_isotropic must pin per_angle_mode to 'individual' (static = "
        "individual everywhere until unification, spec §9); the prior 'constant' "
        "token was inert on the optimizer path and mis-sized reduced-chi2 DOF."
    )
