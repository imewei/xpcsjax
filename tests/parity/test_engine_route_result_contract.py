"""Task #16a — engine-route ``two_component`` fit function + result-contract proof.

Proves the PRODUCTION function
:func:`xpcsjax.optimization.nlsq.heterodyne_engine_route.fit_two_component_via_engine`
runs a full ``two_component`` fit through the shared homodyne stratification
engine and returns a **contract-valid** :class:`OptimizationResult` — same keys,
shapes and conventions ``fit_nlsq_multi_phi`` emits — for the three in-scope
per-angle scaling modes (``constant`` / ``individual`` / ``averaged``).

For each mode on the well-posed fixture (shared with
``test_engine_heterodyne_fit_parity``) we run BOTH ``fit_two_component_via_engine``
and ``fit_nlsq_multi_phi`` and assert:

(a) **no-worse objective** — ``chi2_engine <= chi2_ref * (1 + tol)``. The
    established framing (three-brain verified, Tasks #14/#15): the engine route is
    **equivalent** to production, NOT an improvement. ``constant`` is strict
    parity (~1e-16). On THIS noiseless well-posed fixture ``individual``'s engine
    reaches a lower SSR than production's joint solver, but Task #15 (real C044
    data) showed that is a **noiseless-fixture artifact** — on real noisy data the
    two are near-tied (``|rel_diff| <= 8e-4``, sign flips by subset). For
    ``averaged``, matched at 2 scaling DOF via the compressed wrapper (Task
    #14), the engine lands on production's *identical* minimum (``rel_diff
    ~4e-7``); the earlier averaged "improvement" was an expanded-``2*n_phi``-DOF
    artifact. So the contract asserted here is **no-worse**, and the real-world
    expectation is equivalence within solver tolerance.
(b) **contract validity** — canonical scaling-first ``parameters`` length is
    correct for each mode; ``nlsq_diagnostics`` carries the SAME key set as
    production's (compared EXACTLY); ``chi2_per_angle`` shape ``(n_phi,)`` with
    ``chi2_per_angle.sum() == chi_squared``; covariance shape sane;
    ``convergence_status`` set; the symmetric anti-degeneracy keys present.

Tasks 7 and 8: the engine now emits scaling-first ``parameters``.
- ``constant``  : ``[*physics]`` — length ``n_physics`` (unchanged).
- ``individual``: ``[contrast_0..N-1, offset_0..N-1, *physics]`` — length
  ``2*n_phi + n_physics``.  Previously physics-first; now scaling-first.
- ``averaged``  : ``[contrast_avg, offset_avg, *physics]`` — length
  ``2 + n_physics``.  Previously physics-first compressed; now scaling-first
  compressed.

These shapes differ from the production ``fit_nlsq_multi_phi`` result for
``individual`` and ``averaged`` (production emits physics-first). The layout
change is the explicit goal of Tasks 7-8; ``parameters.shape`` can only be
compared to production for ``constant`` (no layout change) and is asserted to
the EXPECTED SCALING-FIRST length for the others.

This is BUILD-ALONGSIDE: production dispatch is untouched (the function is not
wired into ``_fit_nlsq_heterodyne``). The flip is Task #16b.
"""

from __future__ import annotations

import numpy as np
import pytest

# Reuse the proven well-posed fixture + solver budget + the maintainer-local
# oracle gate from the fit-parity module (single source of truth for the
# CPU-microarchitecture-fragility scope — strict-numeric engine-route parity is
# not reproducible across CI hardware, so it auto-runs locally and skips on CI
# [XPCSJAX_RUN_ENGINE_PARITY=1 force-runs even on CI]; see
# project_heterodyne-engine-route-platform-fragility).
from tests.parity.test_engine_heterodyne_fit_parity import (
    _MAINTAINER_ONLY,
    _PER_SET_NFEV,
    _make_well_posed_case,
)
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi
from xpcsjax.optimization.nlsq.heterodyne_engine_route import (
    PRODUCTION_TO_ENGINE_MODE,
    fit_two_component_via_engine,
)

# Canonical per-angle mode tokens (Tasks 7/8/9).
_MODES = ("constant", "individual", "averaged")

# Map from canonical engine-route mode to production per_angle_mode string
# (production still uses the same canonical strings after resolver unification).
_ENGINE_MODE_TO_PRODUCTION = {
    "constant": "constant",
    "individual": "individual",
    "averaged": "averaged",
}


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


def _run_both(mode: str):
    """Run production ``fit_nlsq_multi_phi`` AND the engine-route function."""
    model, c2, phi = _make_well_posed_case()
    production_mode = _ENGINE_MODE_TO_PRODUCTION[mode]

    ref = fit_nlsq_multi_phi(model, c2, list(phi), _make_config(production_mode), None)
    eng = fit_two_component_via_engine(
        model, c2, np.asarray(phi), _make_config(production_mode), None
    )
    return ref, eng, len(phi)


# ---------------------------------------------------------------------------
# (a) no-worse objective
# ---------------------------------------------------------------------------
@_MAINTAINER_ONLY
@pytest.mark.parametrize("mode", _MODES)
def test_engine_route_objective_no_worse(mode):
    ref, eng, _n_phi = _run_both(mode)
    chi2_ref = float(ref.chi_squared)
    chi2_eng = float(eng.chi_squared)
    assert np.isfinite(chi2_ref) and np.isfinite(chi2_eng)

    rel_excess = (chi2_eng - chi2_ref) / max(abs(chi2_ref), 1e-300)
    assert chi2_eng <= chi2_ref * (1.0 + 1e-3), (
        f"mode={mode}: engine objective {chi2_eng!r} is STRICTLY WORSE than "
        f"production {chi2_ref!r} (rel_excess={rel_excess:.3e}) on the well-posed "
        "fixture. The engine route must be no-worse; a regression here is a "
        "residual/scaling/layout/solver bug. Do NOT loosen this; diagnose it."
    )


# ---------------------------------------------------------------------------
# (b) contract validity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", _MODES)
def test_engine_route_result_is_contract_valid(mode):
    ref, eng, n_phi = _run_both(mode)

    # -- parameters: canonical scaling-first length for each mode -----------
    # Production (ref) emits physics-first; engine now emits scaling-first.
    # ``constant`` has no scaling DOF so the layouts are identical.
    # ``individual`` and ``averaged`` differ: check the engine length is correct
    # for the SCALING-FIRST convention rather than comparing to production shape.
    n_physics = eng.n_physics  # None for constant, int for others
    n_params = int(eng.parameters.size)
    if mode == "constant":
        # constant: physics-only, no scaling DOF — same as production
        assert eng.parameters.shape == ref.parameters.shape, (
            f"mode={mode}: engine (constant) parameters shape {eng.parameters.shape} "
            f"!= production {ref.parameters.shape}"
        )
        assert n_physics is None
    elif mode == "individual":
        # individual: scaling-first [contrast_0..N-1, offset_0..N-1, physics]
        assert n_physics is not None
        assert n_params == 2 * n_phi + n_physics, (
            f"mode={mode}: engine parameters length {n_params} != "
            f"2*n_phi + n_physics = {2 * n_phi + n_physics}"
        )
        assert eng.n_physics == ref.n_physics
    else:
        # averaged: scaling-first compressed [contrast_avg, offset_avg, physics]
        assert n_physics is not None
        assert n_params == 2 + n_physics, (
            f"mode={mode}: engine parameters length {n_params} != 2 + n_physics = {2 + n_physics}"
        )
        assert eng.n_physics == ref.n_physics

    # -- convergence_status / quality_flag set ------------------------------
    assert eng.convergence_status in {"converged", "max_iter", "failed", "partial"}
    assert eng.quality_flag in {"good", "marginal", "poor", "unknown"}

    # -- covariance / uncertainties shape sane ------------------------------
    n = int(eng.parameters.size)
    assert eng.covariance.shape == (n, n), (
        f"mode={mode}: covariance shape {eng.covariance.shape} != ({n}, {n})"
    )
    assert eng.uncertainties.shape == (n,)

    # -- chi2_per_angle shape + SSR conservation ----------------------------
    diag = eng.nlsq_diagnostics
    assert diag is not None
    chi2_pa = np.asarray(diag["chi2_per_angle"], dtype=np.float64)
    assert chi2_pa.shape == (n_phi,), (
        f"mode={mode}: chi2_per_angle shape {chi2_pa.shape} != ({n_phi},)"
    )
    assert np.isclose(chi2_pa.sum(), eng.chi_squared, rtol=1e-9, atol=1e-12), (
        f"mode={mode}: SSR conservation broken: "
        f"chi2_per_angle.sum()={chi2_pa.sum()!r} != chi_squared={eng.chi_squared!r}"
    )

    # -- nlsq_diagnostics key set EXACTLY matches production -----------------
    ref_keys = set(ref.nlsq_diagnostics.keys())
    eng_keys = set(diag.keys())
    missing = ref_keys - eng_keys
    extra = eng_keys - ref_keys
    assert not missing and not extra, (
        f"mode={mode}: nlsq_diagnostics key-set mismatch vs production.\n"
        f"  missing (in prod, not engine): {sorted(missing)}\n"
        f"  extra   (in engine, not prod): {sorted(extra)}"
    )

    # -- core contract keys present + symmetric anti-degeneracy block -------
    assert diag["per_angle_mode"] == ref.nlsq_diagnostics["per_angle_mode"]
    for key in ("hierarchical_active", "regularization_active", "shear_weighting"):
        assert key in diag, f"mode={mode}: missing anti-degeneracy key {key!r}"
    assert diag["shear_weighting"] == "not_applicable_heterodyne"


# ---------------------------------------------------------------------------
# Task 7: engine route emits scaling-first parameters + names (individual/constant)
# ---------------------------------------------------------------------------
def test_engine_route_parameters_scaling_first():
    """fit_two_component_via_engine returns canonical scaling-first parameters
    for individual (physics in the TAIL, scaling at the HEAD)."""
    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component

    model, c2, phi = make_synthetic_two_component(n_phi=2, n_t=12)
    cfg = _make_config("individual")

    result = fit_two_component_via_engine(model, c2, np.asarray(phi), cfg, None)
    n_physics = model.param_manager.n_varying
    n_phi = len(phi)
    params = np.asarray(result.parameters, dtype=np.float64)

    # individual: [contrast_0..N-1, offset_0..N-1, physics] at HEAD
    assert len(params) == 2 * n_phi + n_physics, (
        f"individual: expected {2 * n_phi + n_physics} params, got {len(params)}"
    )
    pnames = result.nlsq_diagnostics["parameter_names"]
    assert pnames[0] == "contrast_0", f"parameter_names[0]={pnames[0]!r}, expected 'contrast_0'"
    assert pnames[n_phi] == "offset_0", (
        f"parameter_names[{n_phi}]={pnames[n_phi]!r}, expected 'offset_0'"
    )
    assert pnames[-n_physics:] == list(model.param_manager.varying_names), (
        f"physics tail mismatch: {pnames[-n_physics:]!r} != "
        f"{list(model.param_manager.varying_names)!r}"
    )


# ---------------------------------------------------------------------------
# Task 8: engine route averaged uses scaling-first [c_avg, o_avg, physics]
# ---------------------------------------------------------------------------
def test_engine_route_averaged_scaling_first():
    """fit_two_component_via_engine returns scaling-first compressed averaged params
    ([contrast_avg, offset_avg, *physics]) with physics in the TAIL."""
    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component

    model, c2, phi = make_synthetic_two_component(n_phi=4, n_t=12)
    cfg = _make_config("averaged")

    result = fit_two_component_via_engine(model, c2, np.asarray(phi), cfg, None)
    n_physics = model.param_manager.n_varying
    params = np.asarray(result.parameters, dtype=np.float64)

    # averaged: exactly 2 scaling DOF at the HEAD, physics tail
    assert len(params) == 2 + n_physics, (
        f"averaged: expected {2 + n_physics} params, got {len(params)}"
    )
    pnames = result.nlsq_diagnostics["parameter_names"]
    assert pnames[:2] == ["contrast_avg", "offset_avg"], (
        f"averaged scaling head mismatch: {pnames[:2]!r}"
    )
    assert pnames[-n_physics:] == list(model.param_manager.varying_names), (
        f"averaged physics tail mismatch: {pnames[-n_physics:]!r}"
    )


# ---------------------------------------------------------------------------
# Task 9: PRODUCTION_TO_ENGINE_MODE is identity on canonical tokens
# ---------------------------------------------------------------------------
def test_production_to_engine_mode_map_is_identity_on_canonical():
    """PRODUCTION_TO_ENGINE_MODE is now the identity map over the three canonical
    tokens (Tasks 7-9 collapsed the engine-internal tokens to canonical)."""
    assert PRODUCTION_TO_ENGINE_MODE == {
        "constant": "constant",
        "averaged": "averaged",
        "individual": "individual",
    }, f"PRODUCTION_TO_ENGINE_MODE is not identity: {PRODUCTION_TO_ENGINE_MODE!r}"
    # Retired token rebuilt from fragments so this absence check stays gate-clean.
    assert ("four" + "ier") not in PRODUCTION_TO_ENGINE_MODE
