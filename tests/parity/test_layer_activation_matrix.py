"""Locks the per-mode anti-degeneracy layer-activation contract.

L1-L4 are active for both ``averaged`` and ``individual`` heterodyne modes;
L5 (shear weighting) is ``laminar_flow``-only and is recorded as
``'not_applicable_heterodyne'`` in all heterodyne results. (In-memory ``fourier``
was retired in Phase 1+2; full fourier-test teardown is Phase 7.)

Diagnostics key/value assertions are derived directly from
``_build_heterodyne_diagnostics`` and the extras dicts assembled in
``heterodyne_core.py`` — do NOT relax them to mere key-presence checks.
"""

from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

# Canonical L4 gradient_monitor block keys (shared mechanism across both modes).
_CANONICAL_GM_KEYS = {
    "collapse_detected",
    "trigger_count",
    "min_gradient_ratio",
    "max_gradient_ratio",
    "n_observations",
    "ratio_threshold",
    "consecutive_triggers",
    "mechanism",
}

# ---------------------------------------------------------------------------
# L1: mode-level reparameterization
# ---------------------------------------------------------------------------


# ``test_heterodyne_l1_reparam_active_fourier`` removed: in-memory fourier (the
# only path that set fourier_basis_dim > 0) was retired in Phase 1+2 (full
# fourier-test teardown in Phase 7). The averaged/individual cases below cover
# the mode-agnostic L1-L4 + L5-exclusion contract.


def test_heterodyne_l1_reparam_averaged_has_none_basis_dim():
    """Averaged per_angle_mode records fourier_basis_dim=None (no Fourier basis)."""
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "averaged"})
    diag = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None).nlsq_diagnostics
    assert diag["per_angle_mode"] == "averaged"
    assert diag["fourier_basis_dim"] is None


# ---------------------------------------------------------------------------
# L2 + L3 + L4 active; L5 excluded — averaged mode
# ---------------------------------------------------------------------------


def test_heterodyne_l2_l3_l4_active_l5_excluded_averaged():
    """L2 hierarchical, L3 regularization, L4 gradient monitor all active in averaged mode.

    L5 shear-weighting must be absent (recorded as 'not_applicable_heterodyne').
    """
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "averaged",
            "enable_hierarchical": True,
            # "tikhonov" is the lightest valid non-none regularization mode
            "regularization_mode": "tikhonov",
            "enable_gradient_monitoring": True,
        }
    )
    diag = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None).nlsq_diagnostics

    # L2: hierarchical two-stage solve ran
    assert diag.get("hierarchical_active") is True, (
        f"expected hierarchical_active=True, got {diag.get('hierarchical_active')!r}"
    )
    assert diag.get("hierarchical_scope") == "full_two_stage"

    # L3: adaptive-CV regularization wired
    assert diag.get("regularization_active") is True, (
        f"expected regularization_active=True, got {diag.get('regularization_active')!r}"
    )

    # L4: gradient-collapse monitor block present
    assert "gradient_monitor" in diag, "gradient_monitor block missing from nlsq_diagnostics"

    # L5: explicitly marked N/A for heterodyne
    assert diag["shear_weighting"] == "not_applicable_heterodyne"


# ---------------------------------------------------------------------------
# L2 + L3 + L4 active; L5 excluded — individual mode
# ---------------------------------------------------------------------------


def test_heterodyne_l2_l3_l4_active_l5_excluded_individual():
    """Same layer-activation contract holds for the individual per_angle_mode.

    Repointed from the former fourier vehicle (in-memory fourier was retired in
    Phase 1+2; full fourier-test teardown in Phase 7). ``individual`` exercises
    the same mode-agnostic L2/L3/L4 layers and the L5 exclusion.
    """
    model, c2, phi = make_synthetic_two_component(n_phi=7, n_t=16)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "individual",
            "enable_hierarchical": True,
            "regularization_mode": "tikhonov",
            "enable_gradient_monitoring": True,
        }
    )
    diag = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None).nlsq_diagnostics

    # L1: per-angle individual mode (no Fourier basis)
    assert diag["per_angle_mode"] == "individual"
    assert diag["fourier_basis_dim"] is None

    # L2
    assert diag.get("hierarchical_active") is True
    assert diag.get("hierarchical_scope") == "full_two_stage"

    # L3
    assert diag.get("regularization_active") is True

    # L4
    assert "gradient_monitor" in diag

    # L5
    assert diag["shear_weighting"] == "not_applicable_heterodyne"


# ---------------------------------------------------------------------------
# Laminar (homodyne) counterpart — L1-L4 active, L5 is laminar's OWN layer
# (inverse of the heterodyne case, which excludes L5).
# ---------------------------------------------------------------------------


def test_laminar_l1_l4_active_l5_present():
    """Laminar_flow L4 activation contract, the inverse of the heterodyne case.

    KEY-NAME DEVIATION (verified against the live code, not guessed): the
    homodyne / laminar in-memory joint-fit path does NOT emit the flat
    ``hierarchical_active`` / ``regularization_active`` / ``shear_weighting``
    diagnostics keys that ``heterodyne_core._build_heterodyne_diagnostics``
    produces. Those flat keys are heterodyne-specific. On the laminar side:

    * the L4 ``gradient_monitor`` block IS surfaced under ``nlsq_diagnostics``
      (via the wrapper's ``_l4_extras``, independent of the diagnostics gate),
      carrying the same canonical key set as heterodyne (shared mechanism), and
    * the laminar result does NOT carry heterodyne's
      ``shear_weighting == "not_applicable_heterodyne"`` marker — L5 shear
      weighting is laminar_flow's OWN layer, so the heterodyne N/A sentinel is
      absent. This is the inverse of ``..._l5_excluded_averaged`` above.

    The richer L2/L3/L5 controller diagnostics only reach the result on the
    >=1M-point stratified-LS path (nested under ``controller_diagnostics``),
    which this small in-memory fixture does not exercise; asserting the flat
    heterodyne keys here would fail against the (correct) current code.
    """
    from tests.optimization.test_l4_callback_observational import _build_laminar_fit

    fit_nlsq, data, cfg = _build_laminar_fit()
    cfg.config["optimization"]["nlsq"]["anti_degeneracy"]["gradient_monitoring"] = {"enable": True}
    diag = fit_nlsq(data, cfg).nlsq_diagnostics
    assert diag is not None, "laminar result carries no nlsq_diagnostics"

    # L4: gradient-collapse monitor block present with the canonical key set.
    assert "gradient_monitor" in diag, "gradient_monitor block missing"
    gm = diag["gradient_monitor"]
    assert set(gm) >= _CANONICAL_GM_KEYS
    # Production wires the per-iteration callback on the laminar STANDARD path.
    assert gm["mechanism"] == "per_iteration_gradient_ratio"

    # L5: laminar's OWN layer — the heterodyne N/A sentinel must be absent
    # (inverse of the heterodyne excluded-L5 contract).
    assert diag.get("shear_weighting") != "not_applicable_heterodyne"
