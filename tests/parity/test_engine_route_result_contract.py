"""Task #16a — engine-route ``two_component`` fit function + result-contract proof.

Proves the PRODUCTION function
:func:`xpcsjax.optimization.nlsq.heterodyne_engine_route.fit_two_component_via_engine`
runs a full ``two_component`` fit through the shared homodyne stratification
engine and returns a **contract-valid** :class:`OptimizationResult` — same keys,
shapes and conventions ``fit_nlsq_multi_phi`` emits — for the three in-scope
per-angle scaling modes (``fixed_constant`` / ``individual`` / ``auto_averaged``).

For each mode on the well-posed fixture (shared with
``test_engine_heterodyne_fit_parity``) we run BOTH ``fit_two_component_via_engine``
and ``fit_nlsq_multi_phi`` and assert:

(a) **no-worse objective** — ``chi2_engine <= chi2_ref * (1 + tol)``. The
    established framing (three-brain verified, Tasks #14/#15): the engine route is
    **equivalent** to production, NOT an improvement. ``fixed_constant`` is strict
    parity (~1e-16). On THIS noiseless well-posed fixture ``individual``'s engine
    reaches a lower SSR than production's joint solver, but Task #15 (real C044
    data) showed that is a **noiseless-fixture artifact** — on real noisy data the
    two are near-tied (``|rel_diff| <= 8e-4``, sign flips by subset). For
    ``auto_averaged``, matched at 2 scaling DOF via the compressed wrapper (Task
    #14), the engine lands on production's *identical* minimum (``rel_diff
    ~4e-7``); the earlier averaged "improvement" was an expanded-``2*n_phi``-DOF
    artifact. So the contract asserted here is **no-worse**, and the real-world
    expectation is equivalence within solver tolerance.
(b) **contract validity** — physics-first ``parameters`` length matches the
    production result; ``nlsq_diagnostics`` carries the SAME key set as
    production's (compared EXACTLY); ``chi2_per_angle`` shape ``(n_phi,)`` with
    ``chi2_per_angle.sum() == chi_squared``; covariance shape sane;
    ``convergence_status`` set; the symmetric anti-degeneracy keys present.

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
    _MODE_TO_PRODUCTION,
    _PER_SET_NFEV,
    _make_well_posed_case,
)
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi
from xpcsjax.optimization.nlsq.heterodyne_engine_route import (
    PRODUCTION_TO_ENGINE_MODE,
    fit_two_component_via_engine,
)

_MODES = ("fixed_constant", "individual", "auto_averaged")


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
    production_mode = _MODE_TO_PRODUCTION[mode]

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

    # -- parameters: physics-first, same length as production ---------------
    assert eng.parameters.shape == ref.parameters.shape, (
        f"mode={mode}: engine parameters shape {eng.parameters.shape} != "
        f"production {ref.parameters.shape}"
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


def test_production_to_engine_mode_map_covers_in_scope():
    """Guard: the production->engine token map covers exactly the resolvable
    in-scope production modes (fourier is intentionally excluded)."""
    assert set(PRODUCTION_TO_ENGINE_MODE) == {"constant", "averaged", "individual"}
    assert set(PRODUCTION_TO_ENGINE_MODE.values()) == {
        "fixed_constant",
        "auto_averaged",
        "individual",
    }


def test_fourier_mode_raises_not_implemented():
    """``fourier`` is out of scope for #16a and must raise (kept on the existing
    path by #16b)."""
    model, c2, phi = _make_well_posed_case()
    cfg = _make_config("fourier")
    with pytest.raises(NotImplementedError, match="fourier"):
        fit_two_component_via_engine(model, c2, np.asarray(phi), cfg, None)
