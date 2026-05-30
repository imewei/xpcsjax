import os
import subprocess
import sys

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
# NOTE: the load-bearing DEFAULT-CI guard is ``test_homodyne_l4_is_diagnostic_only``
# below — it runs unconditionally and asserts monitor-ON == monitor-OFF (bit-
# identical popt + chi2), the observational property. The bit-identity-vs-
# upstream-baseline check (``test_homodyne_characterization_bit_identical_with_monitor``)
# is env-gated and SKIPS in default CI rather than passing vacuously.
# ---------------------------------------------------------------------------


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

    Env-gated and SLOW (~7 min; laminar runs CMA-ES refinement). When
    ``XPCSJAX_RUN_CHARACTERIZATION=1`` is set this runs the characterization
    suite in an isolated subprocess and asserts it stays green (bit-identical
    vs the upstream baseline). When the env var is NOT set we skip EXPLICITLY
    rather than pass vacuously — without the gate the subprocess self-skips and
    returns 0, which would assert nothing about bit-identity. The unconditional
    default-CI guard is ``test_homodyne_l4_is_diagnostic_only``.
    """
    if os.environ.get("XPCSJAX_RUN_CHARACTERIZATION") != "1":
        pytest.skip(
            "characterization is env-gated; XPCSJAX_RUN_CHARACTERIZATION=1 to run. "
            "Default-CI bit-identity guard is test_homodyne_l4_is_diagnostic_only."
        )
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

    assert np.array_equal(
        _homodyne_result_params(r_on), _homodyne_result_params(r_off)
    )
    assert _homodyne_result_chi2(r_on) == _homodyne_result_chi2(r_off)


def test_homodyne_l4_is_per_iteration_block():
    """The laminar result's gradient_monitor block must carry the canonical
    keys, and when per-iteration it must have recorded >= 2 observations."""
    from tests.optimization.test_l4_callback_observational import _build_laminar_fit

    fit_nlsq, data, cfg = _build_laminar_fit()
    cfg.config["optimization"]["nlsq"]["anti_degeneracy"]["gradient_monitoring"] = {
        "enable": True
    }

    res = fit_nlsq(data, cfg)
    gm = _homodyne_gradient_monitor_block(res)
    assert gm["mechanism"] in ("per_iteration_gradient_ratio", "post_solve_fallback")
    assert "collapse_detected" in gm and "max_gradient_ratio" in gm
    if gm["mechanism"] == "per_iteration_gradient_ratio":
        assert gm["n_observations"] >= 2


def test_heterodyne_l4_is_per_iteration_block():
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "averaged",
                                "enable_gradient_monitoring": True})
    res = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)
    gm = res.nlsq_diagnostics["gradient_monitor"]
    assert gm["mechanism"] in ("per_iteration_gradient_ratio", "post_solve_fallback")
    assert "collapse_detected" in gm and "max_gradient_ratio" in gm
    if gm["mechanism"] == "per_iteration_gradient_ratio":
        assert gm["n_observations"] >= 2


def test_heterodyne_l4_is_diagnostic_only():
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    on = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "averaged",
                               "enable_gradient_monitoring": True})
    off = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "averaged",
                                "enable_gradient_monitoring": False})
    r_on = fit_nlsq_multi_phi(model, c2, phi, on, weights=None)
    r_off = fit_nlsq_multi_phi(model, c2, phi, off, weights=None)
    assert np.array_equal(np.asarray(r_on.parameters), np.asarray(r_off.parameters))
    assert r_on.chi_squared == r_off.chi_squared
