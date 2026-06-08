"""Routing tests for the heterodyne standard-tier stratification gate.

The gate (in ``_fit_nlsq_heterodyne``) mirrors homodyne: it *decides* on
stratification above 100k points but only *engages* the stratified-LS solver at
>= 1M points. These tests pin the >=1M boundary by patching the solver and the
point estimator so no large array is ever allocated.
"""

import logging

import numpy as np  # noqa: F401  (kept for parity / future array fixtures)
import pytest


def test_sub_1M_does_not_stratify(monkeypatch):  # noqa: N802 - "1M" pins the >=1M point boundary
    import xpcsjax.optimization.nlsq as nlsq_pkg
    import xpcsjax.optimization.nlsq.heterodyne_stratified_ls as hsl

    called = {"strat": False}
    monkeypatch.setattr(
        hsl,
        "fit_heterodyne_stratified_least_squares",
        lambda **k: called.__setitem__("strat", True),
    )
    from tests.optimization._heterodyne_fixtures import make_cfgmgr_and_data

    cfg, data = make_cfgmgr_and_data(n_phi=3, n_t=20)  # ~1.2k points
    nlsq_pkg.fit_nlsq(data, cfg)
    assert called["strat"] is False


def test_ge_1M_stratifies(monkeypatch):  # noqa: N802 - "1M" pins the >=1M point boundary
    import xpcsjax.optimization.nlsq as nlsq_pkg
    import xpcsjax.optimization.nlsq.heterodyne_stratified_ls as hsl

    sentinel = object()
    called = {"strat": False}

    def _fake(**k):
        called["strat"] = True
        return sentinel

    monkeypatch.setattr(hsl, "fit_heterodyne_stratified_least_squares", _fake)
    # Force point count over 1M without allocating a huge array:
    monkeypatch.setattr(
        "xpcsjax.optimization.nlsq._estimate_heterodyne_points",
        lambda c2, phi: 2_000_000,
    )
    from tests.optimization._heterodyne_fixtures import make_cfgmgr_and_data

    cfg, data = make_cfgmgr_and_data(n_phi=3, n_t=20)
    result = nlsq_pkg.fit_nlsq(data, cfg)
    assert called["strat"] is True
    assert result is sentinel


# =============================================================================
# Task 13 additions: behavioral parity suite for memory/stratification routing
# =============================================================================


# -----------------------------------------------------------------------------
# Part 1 — Parametrized boundary table
#
# Boundary rationale (encoded here to survive future refactors):
#   should_use_stratification() requires n_points > 100_000 to even consider
#   stratifying.  But the stratified-LS SOLVER only engages at >= 1_000_000
#   (homodyne's stratified-LS activation gate, mirrored in _fit_nlsq_heterodyne
#   line: `if use_strat and n_points >= 1_000_000`).  So:
#     - 99_999   → below the "consider" threshold; use_strat=False  → no solver
#     - 100_001  → above "consider"; use_strat may be True, but < 1M → no solver
#     - 999_999  → above "consider"; use_strat may be True, but < 1M → no solver
#     - 1_000_001→ above both thresholds → solver IS called
#
#   The test asserts the SOLVER call, which fires ONLY at >= 1M.
# -----------------------------------------------------------------------------

_BOUNDARY_CASES = [
    pytest.param(99_999, False, id="below-100k-no-solver"),
    pytest.param(100_001, False, id="above-100k-below-1M-no-solver"),
    pytest.param(999_999, False, id="just-below-1M-no-solver"),
    pytest.param(1_000_001, True, id="just-above-1M-solver-fires"),
]


@pytest.mark.parametrize("n_points,expect_solver", _BOUNDARY_CASES)
def test_stratified_ls_boundary(monkeypatch, n_points, expect_solver):
    """Solver fires ONLY at n_points >= 1_000_000.

    All four boundary values exercise the two decision levels:
    (a) should_use_stratification  — threshold > 100k
    (b) the solver gate            — threshold >= 1M
    The test patches only the estimator and the solver; it never allocates
    a large array, so all four cases run cheaply.
    """
    import xpcsjax.optimization.nlsq as nlsq_pkg
    import xpcsjax.optimization.nlsq.heterodyne_stratified_ls as hsl

    sentinel = object()
    called = {"strat": False}

    def _fake_solver(**k):
        called["strat"] = True
        return sentinel

    monkeypatch.setattr(hsl, "fit_heterodyne_stratified_least_squares", _fake_solver)
    monkeypatch.setattr(
        "xpcsjax.optimization.nlsq._estimate_heterodyne_points",
        lambda c2, phi: n_points,
    )

    from tests.optimization._heterodyne_fixtures import make_cfgmgr_and_data

    cfg, data = make_cfgmgr_and_data(n_phi=3, n_t=20)
    result = nlsq_pkg.fit_nlsq(data, cfg)

    assert called["strat"] is expect_solver, (
        f"n_points={n_points}: expected solver_called={expect_solver}, "
        f"got solver_called={called['strat']}"
    )
    if expect_solver:
        assert result is sentinel, "Expected the sentinel returned by the fake solver"


# -----------------------------------------------------------------------------
# Part 2 — Tier-routing: hybrid-streaming takes precedence over stratified-LS
#
# Design choice: patch `select_nlsq_strategy` in the heterodyne_memory module
# (the exact symbol the dispatch imports at runtime — see _fit_nlsq_heterodyne
# lines importing from `xpcsjax.optimization.nlsq.heterodyne_memory`).
# We force strategy=LARGE so the hybrid gate triggers, then assert the
# stratified-LS solver is NOT called.
#
# The hybrid gate also calls build_heterodyne_stratified_data,
# fit_with_stratified_hybrid_streaming_heterodyne, and build_hybrid_streaming_result.
# We patch the top-level streaming fit function (the deepest common call) so we
# don't have to stub the entire chain; build_hybrid_streaming_result is also
# patched because it consumes the streaming fit's output.
#
# Alternative considered and rejected: asserting only the simpler invariant
# "hybrid disabled + >=1M → stratified-LS chosen" — that is already covered
# by test_ge_1M_stratifies above.  The value of Part 2 is testing the
# PRECEDENCE of hybrid over stratified-LS, which is a distinct behavioural
# contract documented in the dispatch comments ("Precedence: cmaes > multi_start
# > hybrid_streaming > local").
# -----------------------------------------------------------------------------


def test_hybrid_streaming_takes_precedence_over_stratified_ls(monkeypatch):
    """LARGE memory tier + hybrid_streaming.enable=true → hybrid path, not stratified-LS.

    Patches:
      - heterodyne_memory.select_nlsq_strategy → forces LARGE tier
      - heterodyne_stratified_data.build_heterodyne_stratified_data → no-op sentinel
      - strategies.heterodyne_hybrid_streaming.fit_with_stratified_hybrid_streaming_heterodyne
            → sentinel (popt/pcov/info triple)
      - heterodyne_result_builder.build_hybrid_streaming_result → sentinel result
      - heterodyne_stratified_ls.fit_heterodyne_stratified_least_squares → sentinel
            (must NOT be called)
    """
    import xpcsjax.optimization.nlsq as nlsq_pkg
    import xpcsjax.optimization.nlsq.heterodyne_memory as het_mem
    import xpcsjax.optimization.nlsq.heterodyne_result_builder as het_rb
    import xpcsjax.optimization.nlsq.heterodyne_stratified_data as het_strat_data
    import xpcsjax.optimization.nlsq.heterodyne_stratified_ls as hsl
    import xpcsjax.optimization.nlsq.strategies.heterodyne_hybrid_streaming as het_hstream
    from xpcsjax.optimization.nlsq.heterodyne_memory import NLSQStrategy, StrategyDecision

    called = {"strat_ls": False, "hybrid": False}

    # Force LARGE tier so the hybrid gate fires.
    def _fake_select(n_points, n_params):
        return StrategyDecision(
            strategy=NLSQStrategy.LARGE,
            threshold_gb=16.0,
            peak_memory_gb=99.0,
            reason="forced-LARGE for test",
        )

    monkeypatch.setattr(het_mem, "select_nlsq_strategy", _fake_select)

    # Stub build_heterodyne_stratified_data → lightweight sentinel object.
    fake_strat = object()
    monkeypatch.setattr(
        het_strat_data,
        "build_heterodyne_stratified_data",
        lambda model, c2, phi, weights=None: fake_strat,
    )

    # Stub the streaming fitter → (popt, pcov, info) triple consumed by result builder.
    import numpy as _np

    fake_popt = _np.zeros(1)
    fake_pcov = _np.zeros((1, 1))
    fake_info = {}

    def _fake_streaming(**k):
        called["hybrid"] = True
        return fake_popt, fake_pcov, fake_info

    monkeypatch.setattr(
        het_hstream,
        "fit_with_stratified_hybrid_streaming_heterodyne",
        _fake_streaming,
    )

    # Stub build_hybrid_streaming_result → sentinel result object.
    hybrid_sentinel = object()
    monkeypatch.setattr(
        het_rb,
        "build_hybrid_streaming_result",
        lambda model, popt, pcov, info, phi_angles: hybrid_sentinel,
    )

    # Stub stratified-LS solver — must NOT be called.
    def _fake_strat_ls(**k):
        called["strat_ls"] = True
        return object()

    monkeypatch.setattr(hsl, "fit_heterodyne_stratified_least_squares", _fake_strat_ls)

    # Also force >=1M points so the stratified-LS gate would fire if hybrid
    # were not taking precedence.
    monkeypatch.setattr(
        "xpcsjax.optimization.nlsq._estimate_heterodyne_points",
        lambda c2, phi: 2_000_000,
    )

    # Build config with hybrid_streaming enabled.
    from tests.optimization._heterodyne_fixtures import make_cfgmgr_and_data

    cfg, data = make_cfgmgr_and_data(
        n_phi=3,
        n_t=20,
        stratification=None,
    )
    # Inject hybrid_streaming.enable into the NLSQ config block so the gate fires.
    cfg.config["optimization"]["nlsq"]["hybrid_streaming"] = {"enable": True}

    result = nlsq_pkg.fit_nlsq(data, cfg)

    assert called["hybrid"] is True, "Hybrid streaming fitter was not called"
    assert called["strat_ls"] is False, (
        "Stratified-LS solver was called despite hybrid_streaming taking precedence"
    )
    assert result is hybrid_sentinel, "Expected the hybrid sentinel result"


# -----------------------------------------------------------------------------
# Task 4 — individual mode is now IN SCOPE for stratified-LS
#
# Explicit `individual` is a JOINT fit (_fit_joint_multi_phi /
# FourierReparameterizer "independent" mode); _aggregate_individual_results is
# only the config-is-None/single-angle fallback. Routing individual through
# stratified-LS is objective-consistent with the in-memory path (no objective
# discontinuity at 1M). Policy: `averaged`/`fourier`/`individual` all route to
# stratified-LS. Only `constant` (frozen scaling) uses the in-memory path.
# -----------------------------------------------------------------------------


def test_individual_mode_uses_stratified_ls(monkeypatch):  # noqa: N802
    """per_angle_mode=individual + >=1M points → stratified-LS solver IS called."""
    import xpcsjax.optimization.nlsq as nlsq_pkg
    import xpcsjax.optimization.nlsq.heterodyne_stratified_ls as hsl

    called = {"strat": False}
    _real = hsl.fit_heterodyne_stratified_least_squares

    def _fake(*, model, c2, phi, config, weights, **k):
        called["strat"] = True
        # Delegate to the real implementation (captured before patching to
        # avoid infinite recursion through the patched name).
        return _real(model=model, c2=c2, phi=phi, config=config, weights=weights, **k)

    monkeypatch.setattr(hsl, "fit_heterodyne_stratified_least_squares", _fake)
    monkeypatch.setattr(
        "xpcsjax.optimization.nlsq._estimate_heterodyne_points",
        lambda c2, phi: 2_000_000,
    )

    from tests.optimization._heterodyne_fixtures import make_cfgmgr_and_data

    cfg, data = make_cfgmgr_and_data(n_phi=3, n_t=20)
    # Request individual explicitly (n_phi=3 would auto-resolve to averaged).
    cfg.config["optimization"]["nlsq"]["per_angle_mode"] = "individual"

    nlsq_pkg.fit_nlsq(data, cfg)
    assert called["strat"] is True, (
        "Stratified-LS solver was NOT called for individual mode at >=1M (should use it)"
    )


def test_ge_1M_unsupported_mode_warns(monkeypatch, caplog):  # noqa: N802 - "1M" pins the boundary
    """>=1M points in an UNSUPPORTED per-angle mode (constant) → WARNING, not silence.

    "No silent caps": a fit large enough that stratification would have mattered
    (>=1M) that is routed to the higher-memory in-memory joint fit ONLY because
    its mode lacks a stratified expander must say so at WARNING level. The fit
    must still complete (fall back to the in-memory joint fit) and return a valid
    OptimizationResult.

    ``constant`` is now the only remaining mode that lacks a stratified expander
    (``individual`` became supported in Task 4).
    """
    import xpcsjax.optimization.nlsq as nlsq_pkg
    import xpcsjax.optimization.nlsq.heterodyne_stratified_ls as hsl

    # The stratified-LS solver must NOT be called (constant is unsupported).
    called = {"strat": False}

    def _fake(**k):
        called["strat"] = True
        return object()

    monkeypatch.setattr(hsl, "fit_heterodyne_stratified_least_squares", _fake)
    # Force the gate above 1M without allocating a huge array.
    monkeypatch.setattr(
        "xpcsjax.optimization.nlsq._estimate_heterodyne_points",
        lambda c2, phi: 2_000_000,
    )

    from tests.optimization._heterodyne_fixtures import make_cfgmgr_and_data

    cfg, data = make_cfgmgr_and_data(n_phi=3, n_t=20)
    # ``constant`` is the remaining mode unsupported by stratified-LS.
    cfg.config["optimization"]["nlsq"]["per_angle_mode"] = "constant"

    with caplog.at_level(logging.WARNING, logger="xpcsjax.optimization.nlsq"):
        result = nlsq_pkg.fit_nlsq(data, cfg)

    # The stratified-LS solver was correctly skipped...
    assert called["strat"] is False
    # ...and the fit still produced a valid result (fell back to the joint fit).
    assert result is not None
    assert hasattr(result, "parameters")

    # A WARNING-level record naming the mode + the skip must have been emitted.
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    matching = [
        r
        for r in warnings
        if "constant" in r.getMessage() and "stratified-LS skipped" in r.getMessage()
    ]
    assert matching, (
        "Expected a WARNING that names per_angle_mode=constant and reports "
        f"stratified-LS was skipped; got warnings: {[r.getMessage() for r in warnings]}"
    )


# -----------------------------------------------------------------------------
# Fix 2 — flat enable_cmaes takes precedence over the stratified-LS gate
#
# Config supports the FLAT `optimization.nlsq.enable_cmaes: true` field (parsed
# into NLSQConfig.enable_cmaes). When CMA-ES is on, the stratified-LS gate must
# be skipped so fit_nlsq_multi_phi can delegate to CMA-ES.
# -----------------------------------------------------------------------------


def test_flat_enable_cmaes_skips_stratified_ls(monkeypatch):  # noqa: N802
    """Flat enable_cmaes=true (no nested cmaes block) + >=1M → stratified-LS NOT called."""
    import xpcsjax.optimization.nlsq as nlsq_pkg
    import xpcsjax.optimization.nlsq.heterodyne_stratified_ls as hsl
    from xpcsjax.optimization.nlsq.heterodyne_config import (
        NLSQConfig as _HetNLSQConfig,
    )

    called = {"strat": False}

    def _fake(**k):
        called["strat"] = True
        return object()

    monkeypatch.setattr(hsl, "fit_heterodyne_stratified_least_squares", _fake)
    monkeypatch.setattr(
        "xpcsjax.optimization.nlsq._estimate_heterodyne_points",
        lambda c2, phi: 2_000_000,
    )

    from tests.optimization._heterodyne_fixtures import make_cfgmgr_and_data

    cfg, data = make_cfgmgr_and_data(n_phi=3, n_t=20)
    # Flat enable_cmaes, NO nested cmaes block.
    cfg.config["optimization"]["nlsq"]["enable_cmaes"] = True

    # Verify the parsed config actually carries enable_cmaes=True from the flat field.
    parsed = _HetNLSQConfig.from_dict(dict(cfg.config["optimization"]["nlsq"]))
    assert parsed.enable_cmaes is True

    nlsq_pkg.fit_nlsq(data, cfg)
    assert called["strat"] is False, (
        "Stratified-LS solver was called despite flat enable_cmaes=true (CMA-ES precedence)"
    )
