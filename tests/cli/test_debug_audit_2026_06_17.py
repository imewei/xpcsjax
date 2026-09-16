"""Regression tests for the 2026-06-17 debug-audit fixes (CLI layer)."""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# _config_summary must read xpcsjax's real ConfigManager surface
# (config["analysis_mode"], .config_file, data_type), not the nonexistent
# mode/data_type/config_path attributes (finding #4).
# ---------------------------------------------------------------------------
def test_config_summary_reads_real_surface() -> None:
    from xpcsjax.cli.result_saving import _config_summary
    from xpcsjax.config.manager import ConfigManager

    cm = ConfigManager.__new__(ConfigManager)
    cm.config = {
        "analysis_mode": "laminar_flow",
        "experimental_data": {"data_type": "aps_u"},
    }
    cm.config_file = "/path/to/config.yaml"

    summary = _config_summary(cm)
    assert summary["mode"] == "laminar_flow"
    assert summary["data_type"] == "aps_u"
    assert summary["config_path"] == "/path/to/config.yaml"


# ---------------------------------------------------------------------------
# save_results_npz must store NaN arrays at the documented (n,)/(n,n) shapes
# when uncertainties/covariance are None, not a 0-d scalar (finding #19).
# ---------------------------------------------------------------------------
def test_save_npz_none_covariance_uses_documented_shapes(tmp_path) -> None:
    from xpcsjax.cli.result_saving import save_results_npz
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    # A non-converged / global-escape result legitimately carries None
    # covariance and uncertainties.
    result = OptimizationResult(
        parameters=np.array([1.0, 2.0, 3.0]),
        uncertainties=None,  # type: ignore[arg-type]
        covariance=None,  # type: ignore[arg-type]
        chi_squared=1.0,
        reduced_chi_squared=0.5,
        convergence_status="max_iter",
        iterations=0,
        execution_time=0.1,
        device_info={},
    )
    out = save_results_npz(result, tmp_path)
    with np.load(out) as data:
        assert data["uncertainties"].shape == (3,)
        assert data["covariance"].shape == (3, 3)
        assert np.all(np.isnan(data["uncertainties"]))
        assert np.all(np.isnan(data["covariance"]))


# ---------------------------------------------------------------------------
# --tolerance must also write gtol so it can relax the gradient-norm criterion
# (finding #20).
#
# The dead `optimization_runner.apply_cli_overrides` this test used to exercise
# was deleted (audit B8: it had zero production callers -- run_nlsq calls
# service.fit.run_fit directly). The same ftol/xtol/gtol-together behavior is
# now covered directly against the service function it actually goes through:
# see tests/service/test_fit_overrides.py::test_tolerance_relaxes_ftol_xtol_and_gtol_together.
# ---------------------------------------------------------------------------
