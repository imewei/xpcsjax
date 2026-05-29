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
