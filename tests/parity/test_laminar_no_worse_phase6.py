"""Phase 6 no-worse-SSR gate: deleting fourier on the laminar paths does not degrade the
resolved individual/averaged/constant SSR. Synthetic + self-contained — runs in CI.

Laminar layout is already scaling-first, so fourier deletion is pure dead-code removal and
the resolved-mode solves must be unchanged (the individual tripwire below is the rtol-class
guard; the maintainer-local homodyne goldens enforce the full rtol=1e-10 explicit-individual
contract).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore:Ill-conditioned Jacobian")

_NAMES = ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]


def _homodyne_config(n_t: int) -> dict:
    return {
        "analysis_mode": "laminar_flow",
        "analyzer_parameters": {
            "dt": 0.1,
            "start_frame": 1,
            "end_frame": n_t,
            "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": n_t},
            "scattering": {"wavevector_q": 0.0237},
            "geometry": {"stator_rotor_gap": 2_000_000.0},
        },
    }


def _stratified_info(per_angle_mode: str) -> dict:
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.homodyne_model import HomodyneModel
    from xpcsjax.optimization.nlsq.strategies.stratified_ls import (
        fit_with_stratified_least_squares,
    )

    phi = np.array([0.0, 36.0, 72.0, 108.0, 144.0], dtype=np.float64)
    n_t = 12
    t = np.linspace(0.0, float(n_t - 1), n_t, dtype=np.float64)
    truth = np.array([1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0])
    t1, t2 = np.meshgrid(t, t, indexing="ij")

    cfg = ConfigManager(config_override=_homodyne_config(n_t))
    model = HomodyneModel(cfg.config)
    # compute_c2 returns (n_phi, n_t, n_t); use as the synthetic g2 surfaces.
    g2 = np.asarray(model.compute_c2(truth, phi, contrast=0.3, offset=1.0), dtype=np.float64)

    n_phi = len(phi)
    strat = SimpleNamespace(
        phi_flat=np.repeat(phi, n_t * n_t),
        t1_flat=np.tile(t1.ravel(), n_phi),
        t2_flat=np.tile(t2.ravel(), n_phi),
        g2_flat=g2.ravel(),
        sigma=None,
        q=0.0237,
        L=2_000_000.0,
        dt=0.1,
        stratification_diagnostics=None,
        chunk_sizes=None,
    )
    # Warm-start physics at the truth (self-consistent g2 -> residuals ~0). Starting
    # physics at all-zeros collapses D0=0 -> zero gradient on the diffusion-only term.
    x0 = np.concatenate([np.full(n_phi, 0.3), np.full(n_phi, 1.0), truth])
    lo = np.concatenate(
        [np.zeros(n_phi), np.full(n_phi, 0.5), np.array([1e-6, -2.0, -50.0, -1.0, -2.0, -50.0, -180.0])]
    )
    hi = np.concatenate(
        [np.ones(n_phi), np.full(n_phi, 1.5), np.array([1e5, 2.0, 50.0, 1.0, 2.0, 50.0, 180.0])]
    )
    _popt, _pcov, info = fit_with_stratified_least_squares(
        stratified_data=strat,
        per_angle_scaling=True,
        physical_param_names=_NAMES,
        initial_params=x0,
        bounds=(lo, hi),
        log=logging.getLogger("t7"),
        anti_degeneracy_config={"enable": True, "per_angle_mode": per_angle_mode},
        analysis_mode="laminar_flow",
    )
    return info


def _ssr(info: dict) -> float:
    # final_cost is the data-only SSR (see test_laminar_execute_layers docstring)
    return float(info["final_cost"])


@pytest.mark.parametrize("mode", ["individual", "auto", "constant"])
def test_stratified_ssr_finite(mode):
    ssr = _ssr(_stratified_info(mode))
    assert np.isfinite(ssr) and ssr >= 0.0


def test_stratified_individual_ssr_tripwire():
    """Self-consistent g2-at-truth drives residuals -> ~0; the individual solve must reach it.

    A drift above 1e-6 means the fourier deletion perturbed the reached individual solve —
    impossible if the deletion was dead-code-only (laminar is already scaling-first).
    """
    ssr = _ssr(_stratified_info("individual"))
    assert ssr < 1e-6, f"individual SSR drifted: {ssr}"


def test_fourier_rejected_all_laminar_paths():
    for mode in ("fourier", "independent"):
        with pytest.raises(ValueError, match="per_angle_mode"):
            _stratified_info(mode)
