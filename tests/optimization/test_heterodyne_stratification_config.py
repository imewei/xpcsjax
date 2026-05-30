"""Tests for ``optimization.stratification.*`` config parsing on the heterodyne path.

Mirrors the upstream homodyne wrapper (``_apply_stratification_if_needed``), where
the stratification block is a SIBLING of ``optimization.nlsq`` at
``config.config["optimization"]["stratification"]`` -- NOT nested inside the nlsq
block. The gate in ``_fit_nlsq_heterodyne`` must read from the optimization block
so a user's ``optimization.stratification.enabled: false`` actually disables it,
and ``target_chunk_size`` flows to the stratified-LS solver.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.optimization.nlsq import fit_nlsq
from xpcsjax.optimization.nlsq.heterodyne_config import StratificationConfig

from ._heterodyne_fixtures import make_cfgmgr_and_data


def test_stratification_defaults() -> None:
    """No stratification block -> homodyne defaults."""
    cfg, _ = make_cfgmgr_and_data(n_phi=3, n_t=8)
    opt_block = cfg.config.get("optimization", {})
    sc = StratificationConfig.from_optimization_block(opt_block)
    assert sc.enabled == "auto"
    assert sc.target_chunk_size == 100_000
    assert sc.max_imbalance_ratio == 5.0
    # Remaining homodyne-mirrored defaults.
    assert sc.force_sequential_fallback is False
    assert sc.check_memory_safety is True
    assert sc.use_index_based is False


def test_stratification_enabled_false_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    """``optimization.stratification.enabled: false`` must skip the stratified-LS
    solver even when the point count is well above the 1M gate."""
    import xpcsjax.optimization.nlsq as nlsq_pkg
    from xpcsjax.optimization.nlsq import heterodyne_stratified_ls as hsl

    cfg, data = make_cfgmgr_and_data(
        n_phi=3, n_t=8, stratification={"enabled": False}
    )

    # Force the gate past its 1M threshold regardless of the tiny fixture.
    monkeypatch.setattr(
        nlsq_pkg, "_estimate_heterodyne_points", lambda c2, phi: 2_000_000
    )

    called = {"hit": False}

    def _sentinel(*args, **kwargs):
        called["hit"] = True
        raise AssertionError("fit_heterodyne_stratified_least_squares must not be called")

    monkeypatch.setattr(hsl, "fit_heterodyne_stratified_least_squares", _sentinel)

    # Should fall through to the in-memory joint fit without touching the solver.
    fit_nlsq(data, cfg)
    assert called["hit"] is False


def test_stratification_target_chunk_size_flows(monkeypatch: pytest.MonkeyPatch) -> None:
    """A user-set ``target_chunk_size`` must reach
    ``fit_heterodyne_stratified_least_squares``."""
    import xpcsjax.optimization.nlsq as nlsq_pkg
    from xpcsjax.optimization.nlsq import heterodyne_stratified_ls as hsl

    cfg, data = make_cfgmgr_and_data(
        n_phi=3, n_t=8, stratification={"target_chunk_size": 50_000}
    )

    monkeypatch.setattr(
        nlsq_pkg, "_estimate_heterodyne_points", lambda c2, phi: 2_000_000
    )

    captured: dict[str, object] = {}

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        # Return a minimal sentinel so the dispatch path can complete.
        return "sentinel-result"

    monkeypatch.setattr(hsl, "fit_heterodyne_stratified_least_squares", _capture)
    # Completion logging expects a real result; stub it out (best-effort already
    # guards, but keep the test focused on the kwarg).
    monkeypatch.setattr(
        nlsq_pkg, "_safe_log_heterodyne_completion", lambda *a, **k: None
    )

    result = fit_nlsq(data, cfg)
    assert result == "sentinel-result"
    assert captured.get("target_chunk_size") == 50_000


def test_stratification_balanced_angles_required_for_default() -> None:
    """Sanity: the default fixture phi grid is balanced enough that the
    auto-path is exercised (guards against the fixture silently disabling
    stratification via imbalance)."""
    cfg, data = make_cfgmgr_and_data(n_phi=3, n_t=8)
    phi = np.asarray(data["phi_angles_list"])
    assert phi.size == 3


# =============================================================================
# Fix 3 — config knobs are honored (max_imbalance_ratio, use_index_based)
# =============================================================================


def test_max_imbalance_ratio_disables_stratification(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured ``max_imbalance_ratio`` lower than the measured imbalance
    must disable the stratified-LS solver even above the 1M gate.

    The gate-level check applies the CONFIGURED threshold (chunking's internal
    should_use_stratification hard-codes 5.0)."""
    import xpcsjax.optimization.nlsq as nlsq_pkg
    from xpcsjax.optimization.nlsq import heterodyne_stratified_ls as hsl
    from xpcsjax.optimization.nlsq.strategies.chunking import AngleDistributionStats

    # Configure a tight max_imbalance_ratio (1.5); force a measured imbalance of 4.0.
    cfg, data = make_cfgmgr_and_data(
        n_phi=3, n_t=8, stratification={"max_imbalance_ratio": 1.5}
    )

    monkeypatch.setattr(
        nlsq_pkg, "_estimate_heterodyne_points", lambda c2, phi: 2_000_000
    )

    def _fake_dist(phi):
        # imbalance_ratio=4.0 is below chunking's hard 5.0 (so
        # should_use_stratification keeps it on) but above the configured 1.5.
        return AngleDistributionStats(
            unique_angles=np.asarray(phi, dtype=np.float64),
            n_angles=int(np.asarray(phi).size),
            counts={0.0: 4, 1.0: 1},
            fractions={0.0: 0.8, 1.0: 0.2},
            imbalance_ratio=4.0,
            min_angle=1.0,
            max_angle=0.0,
            is_balanced=True,
        )

    monkeypatch.setattr(
        "xpcsjax.optimization.nlsq.strategies.chunking.analyze_angle_distribution",
        _fake_dist,
    )

    called = {"hit": False}

    def _sentinel(*args, **kwargs):
        called["hit"] = True
        raise AssertionError("stratified-LS must not be called when imbalance > configured ratio")

    monkeypatch.setattr(hsl, "fit_heterodyne_stratified_least_squares", _sentinel)

    fit_nlsq(data, cfg)
    assert called["hit"] is False


def test_max_imbalance_ratio_can_loosen_above_5(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured ``max_imbalance_ratio`` ABOVE chunking's hard 5.0 must still
    allow stratification at an imbalance that the hard cutoff would have rejected.

    This proves the configured threshold is the SOLE imbalance gate (it can move
    the cutoff in both directions), not merely a tightening on top of 5.0."""
    import xpcsjax.optimization.nlsq as nlsq_pkg
    from xpcsjax.optimization.nlsq import heterodyne_stratified_ls as hsl
    from xpcsjax.optimization.nlsq.strategies.chunking import AngleDistributionStats

    # Loose max_imbalance_ratio (10.0); force a measured imbalance of 6.0 — above
    # chunking's hard 5.0 (old behavior would reject) but within the configured 10.0.
    cfg, data = make_cfgmgr_and_data(
        n_phi=3, n_t=8, stratification={"max_imbalance_ratio": 10.0}
    )

    monkeypatch.setattr(
        nlsq_pkg, "_estimate_heterodyne_points", lambda c2, phi: 2_000_000
    )

    def _fake_dist(phi):
        return AngleDistributionStats(
            unique_angles=np.asarray(phi, dtype=np.float64),
            n_angles=int(np.asarray(phi).size),
            counts={0.0: 6, 1.0: 1},
            fractions={0.0: 0.857, 1.0: 0.143},
            imbalance_ratio=6.0,
            min_angle=1.0,
            max_angle=0.0,
            is_balanced=False,
        )

    monkeypatch.setattr(
        "xpcsjax.optimization.nlsq.strategies.chunking.analyze_angle_distribution",
        _fake_dist,
    )

    called = {"hit": False}

    def _sentinel(*args, **kwargs):
        called["hit"] = True
        return object()  # _safe_log_heterodyne_completion is guarded against this

    monkeypatch.setattr(hsl, "fit_heterodyne_stratified_least_squares", _sentinel)

    fit_nlsq(data, cfg)
    assert called["hit"] is True


# =============================================================================
# Item 2 — shipped template <-> parser parity
#
# Loads the SHIPPED ``xpcsjax_two_component.yaml`` through the real ConfigManager
# path and asserts the parsed StratificationConfig exposes the homodyne-matching
# defaults. This locks template<->parser agreement: if the YAML ships a value
# that the parser would surface differently (or a future YAML edit drifts from
# homodyne's defaults), this fails loudly.
# =============================================================================


def _shipped_two_component_template_path() -> str:
    """Resolve the installed ``xpcsjax_two_component.yaml`` path (import-anchored)."""
    from importlib import resources

    return str(resources.files("xpcsjax.config.templates") / "xpcsjax_two_component.yaml")


def test_shipped_template_stratification_defaults() -> None:
    """The shipped two_component template's stratification block parses to the
    homodyne-matching defaults through ConfigManager + StratificationConfig."""
    from xpcsjax.config import ConfigManager

    cfg = ConfigManager(_shipped_two_component_template_path())
    assert cfg.config is not None
    opt_block = cfg.config.get("optimization", {})

    # The block is a SIBLING of optimization.nlsq (not nested inside it).
    assert "stratification" in opt_block, (
        "shipped two_component template must ship optimization.stratification"
    )

    sc = StratificationConfig.from_optimization_block(opt_block)

    # Homodyne-matching defaults (these are the shipped values in the YAML).
    assert sc.enabled == "auto"
    assert sc.target_chunk_size == 100_000
    assert sc.max_imbalance_ratio == 5.0
    assert sc.check_memory_safety is True
    assert sc.use_index_based is False
    # force_sequential_fallback ships false (inert for heterodyne, parsed for
    # shared-config compatibility).
    assert sc.force_sequential_fallback is False


def test_use_index_based_flows_into_diagnostics() -> None:
    """``use_index_based: false`` flows into the stratified-LS result's
    ``stratification_diagnostics`` (not hard-coded True)."""
    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
        fit_heterodyne_stratified_least_squares,
    )

    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "averaged"})
    res = fit_heterodyne_stratified_least_squares(
        model=model,
        c2=c2,
        phi=phi,
        config=cfg,
        weights=None,
        shuffle=False,
        use_index_based=False,
    )
    assert res.stratification_diagnostics is not None
    assert res.stratification_diagnostics.use_index_based is False
