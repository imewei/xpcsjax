import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

# ---------------------------------------------------------------------------
# Homodyne (laminar_flow) — Task 3.
#
# The hard safety property: wiring the per-iteration L4 monitor into the live
# laminar_flow solve must leave the rtol=1e-10 homodyne characterization
# baseline BIT-IDENTICAL. The monitor is strictly observational.
#
# NOTE: the load-bearing, dependency-free guard is ``test_homodyne_l4_is_diagnostic_only``
# below — it runs unconditionally and asserts monitor-ON == monitor-OFF (bit-
# identical popt + chi2), the observational property. The bit-identity-vs-
# upstream-baseline check (``test_homodyne_characterization_bit_identical_with_monitor``)
# is AVAILABILITY-gated, not env-gated: it RUNS whenever the upstream ``homodyne``
# package and the maintainer datasets are present (no env var needed), and SKIPS
# with a clear reason when either is absent — so it executes on a maintainer
# machine and stays green (skipped, never failing) on CI / fresh clones.
# ---------------------------------------------------------------------------

# Datasets the characterization subprocess reads (paths mirror
# tests/characterization/test_homodyne_equivalence.py CONFIGS — keep in sync).
_CHARACTERIZATION_CONFIGS = (
    Path("/home/wei/Documents/Projects/data/Simon/homodyne_static_config.yaml"),
    Path("/home/wei/Documents/Projects/data/C020/homodyne_laminar_flow_config.yaml"),
)


def _characterization_available() -> tuple[bool, str]:
    """Return ``(available, reason_if_not)`` for the upstream-baseline check.

    Available iff the upstream ``homodyne`` package is importable AND every
    characterization dataset config exists. ``find_spec`` is used so we probe
    importability without importing (and triggering) the package.
    """
    if importlib.util.find_spec("homodyne") is None:
        return False, "upstream `homodyne` package not importable"
    missing = [str(p) for p in _CHARACTERIZATION_CONFIGS if not p.exists()]
    if missing:
        return False, f"characterization datasets absent: {missing}"
    return True, ""


def _homodyne_gradient_monitor_block(result):
    """Pull the L4 ``gradient_monitor`` block from a homodyne result.

    The block lives under ``nlsq_diagnostics`` (the same key heterodyne uses).
    """
    diag = getattr(result, "nlsq_diagnostics", None)
    assert diag is not None, "homodyne result carries no nlsq_diagnostics"
    assert "gradient_monitor" in diag, "no gradient_monitor block in nlsq_diagnostics"
    return diag["gradient_monitor"]


def _homodyne_result_params(result):
    for attr in ("parameters", "popt", "x"):
        if hasattr(result, attr):
            return np.asarray(getattr(result, attr), dtype=np.float64)
    raise AttributeError("no parameter attribute on result")


def _homodyne_result_chi2(result):
    for attr in ("chi_squared", "chi2", "final_cost", "cost"):
        if hasattr(result, attr):
            return getattr(result, attr)
    raise AttributeError("no chi-squared attribute on result")


def test_homodyne_characterization_bit_identical_with_monitor():
    """Hard safety gate: the homodyne rtol=1e-10 characterization suite must
    still pass with the per-iteration L4 monitor wired in.

    SLOW (~7 min; laminar runs CMA-ES refinement). Runs the characterization
    suite in an isolated subprocess and asserts it stays green (bit-identical
    vs the upstream baseline). The child suite is itself env-gated, so the
    subprocess env forces ``XPCSJAX_RUN_CHARACTERIZATION=1`` — without it the
    child self-skips and returns 0, asserting nothing about bit-identity.

    AVAILABILITY-gated (not env-gated): it RUNS when the upstream ``homodyne``
    package and the maintainer datasets are present, and SKIPS with a clear
    reason otherwise — so it executes on a maintainer machine without an env var
    and stays green (skipped) on CI / fresh clones. The dependency-free
    observational guard is ``test_homodyne_l4_is_diagnostic_only``.
    """
    available, reason = _characterization_available()
    if not available:
        pytest.skip(f"upstream-baseline characterization unavailable: {reason}")

    # Force the characterization gate ON in the child so it actually runs
    # (the child self-skips and returns 0 — a vacuous pass — without it).
    child_env = {**os.environ, "XPCSJAX_RUN_CHARACTERIZATION": "1"}
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/characterization/test_homodyne_equivalence.py",
            "-q",
        ],
        capture_output=True,
        text=True,
        env=child_env,
    )
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


def test_homodyne_l4_is_diagnostic_only():
    """Monitoring ON vs OFF must produce bit-identical popt and chi2."""
    from tests.optimization.test_l4_callback_observational import _build_laminar_fit

    fit_nlsq, data, cfg_on = _build_laminar_fit()
    # _build_laminar_fit disables anti_degeneracy; re-enable just the L4
    # gradient-monitoring gate for the ON run, leaving every other layer off.
    cfg_on.config["optimization"]["nlsq"]["anti_degeneracy"]["gradient_monitoring"] = {
        "enable": True
    }

    _, data_off, cfg_off = _build_laminar_fit()
    cfg_off.config["optimization"]["nlsq"]["anti_degeneracy"]["gradient_monitoring"] = {
        "enable": False
    }

    r_on = fit_nlsq(data, cfg_on)
    r_off = fit_nlsq(data_off, cfg_off)

    assert np.array_equal(_homodyne_result_params(r_on), _homodyne_result_params(r_off))
    assert _homodyne_result_chi2(r_on) == _homodyne_result_chi2(r_off)
    # Covariance (pcov) bit-identity is part of the same hard gate as popt + chi2.
    cov_on = getattr(r_on, "covariance", None)
    cov_off = getattr(r_off, "covariance", None)
    if cov_on is not None and cov_off is not None:
        assert np.array_equal(np.asarray(cov_on), np.asarray(cov_off))


def test_homodyne_l4_is_per_iteration_block():
    """The laminar result's gradient_monitor block must carry the canonical
    keys, and when per-iteration it must have recorded >= 2 observations."""
    from tests.optimization.test_l4_callback_observational import _build_laminar_fit

    fit_nlsq, data, cfg = _build_laminar_fit()
    cfg.config["optimization"]["nlsq"]["anti_degeneracy"]["gradient_monitoring"] = {"enable": True}

    res = fit_nlsq(data, cfg)
    gm = _homodyne_gradient_monitor_block(res)
    assert "collapse_detected" in gm and "max_gradient_ratio" in gm
    # The laminar STANDARD curve_fit path wires the per-iteration callback
    # explicitly (Phase-0 seam), so the live mechanism MUST be
    # per_iteration_gradient_ratio. Strict assertion catches a silent regression
    # to the post-solve covariance-condition fallback.
    assert gm["mechanism"] == "per_iteration_gradient_ratio"
    assert gm["n_observations"] >= 2


def test_heterodyne_l4_is_per_iteration_block():
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "averaged",
            "enable_gradient_monitoring": True,
        }
    )
    res = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)
    gm = res.nlsq_diagnostics["gradient_monitor"]
    assert "collapse_detected" in gm and "max_gradient_ratio" in gm
    # Production wires the per-iteration callback explicitly on the heterodyne
    # joint-fit path, so the live mechanism MUST be per_iteration_gradient_ratio
    # (empirically n_observations ~= 782 for this fixture). Asserting strictly
    # here means a future regression that silently degrades the joint-fit path to
    # the post-solve covariance-condition fallback fails CI instead of passing
    # vacuously through a permissive `mechanism in (...)` check.
    assert gm["mechanism"] == "per_iteration_gradient_ratio"
    assert gm["n_observations"] >= 2


def test_heterodyne_l4_is_diagnostic_only():
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    on = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "averaged",
            "enable_gradient_monitoring": True,
        }
    )
    off = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "averaged",
            "enable_gradient_monitoring": False,
        }
    )
    r_on = fit_nlsq_multi_phi(model, c2, phi, on, weights=None)
    r_off = fit_nlsq_multi_phi(model, c2, phi, off, weights=None)
    assert np.array_equal(np.asarray(r_on.parameters), np.asarray(r_off.parameters))
    assert r_on.chi_squared == r_off.chi_squared
    # Covariance (pcov) bit-identity is part of the same hard gate as popt + chi2.
    cov_on = getattr(r_on, "covariance", None)
    cov_off = getattr(r_off, "covariance", None)
    if cov_on is not None and cov_off is not None:
        assert np.array_equal(np.asarray(cov_on), np.asarray(cov_off))


# ---------------------------------------------------------------------------
# Task 4 additions: cross-mode block-key parity + fallback coverage
# ---------------------------------------------------------------------------

_EXPECTED_GM_KEYS = {
    "collapse_detected",
    "trigger_count",
    "min_gradient_ratio",
    "max_gradient_ratio",
    "n_observations",
    "ratio_threshold",
    "consecutive_triggers",
    "mechanism",
}


def test_both_modes_emit_same_l4_block_keys():
    """Heterodyne AND laminar_flow must emit the SAME canonical gradient_monitor
    key set — L4 is a shared mechanism, so the block contract is mode-agnostic.

    Heterodyne side locks to the canonical keys; the laminar side is fitted with
    monitoring enabled and its block-key set is asserted EQUAL to the heterodyne
    block's, so a future divergence in either result builder fails CI.
    """
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    het = fit_nlsq_multi_phi(
        model,
        c2,
        phi,
        NLSQConfig.from_dict(
            {
                "analysis_mode": "two_component",
                "per_angle_mode": "averaged",
                "enable_gradient_monitoring": True,
            }
        ),
        weights=None,
    )
    het_keys = set(het.nlsq_diagnostics["gradient_monitor"])
    assert het_keys >= _EXPECTED_GM_KEYS

    # Laminar counterpart: same shared L4 block, asserted to carry the SAME keys.
    from tests.optimization.test_l4_callback_observational import _build_laminar_fit

    fit_nlsq, data, cfg = _build_laminar_fit()
    cfg.config["optimization"]["nlsq"]["anti_degeneracy"]["gradient_monitoring"] = {"enable": True}
    lam = fit_nlsq(data, cfg)
    lam_keys = set(lam.nlsq_diagnostics["gradient_monitor"])
    assert lam_keys == het_keys


def test_l4_fallback_block_when_no_observations():
    """An empty monitor (callback never fired) must produce mechanism='post_solve_fallback'."""
    from xpcsjax.optimization.nlsq.gradient_monitor import (
        GradientCollapseMonitor,
        GradientMonitorConfig,
        gradient_monitor_diagnostics,
    )

    mon = GradientCollapseMonitor(
        GradientMonitorConfig(), physical_indices=[0], per_angle_indices=[1]
    )
    block = gradient_monitor_diagnostics(mon)  # empty history
    assert block["mechanism"] == "post_solve_fallback"
    assert "collapse_detected" in block
    assert "max_gradient_ratio" in block
