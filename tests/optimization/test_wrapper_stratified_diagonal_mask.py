"""The wrapper's per-point (full-copy) stratified model branch must not fit the
diagonal.

Real data is diagonal-corrected by the loader (the zero-lag self-correlation
spike is replaced by neighbor interpolation); the theory value there is
``offset + contrast``, which matches neither the spike nor the interpolation.
The non-stratified branch applies ``apply_diagonal_correction`` to the theory
grid and the engine residual (``strategies/residual_jit.py``) masks ``t1 == t2``;
the stratified per-point branch did neither, so the optimizer distorted the
physics to chase ~n_phi*n_t lag-free points (120k-point synthetic: SSR_offdiag
5.45 vs 0.03 on the standard path, D0 1319 vs 1000).

Pin: on identical diagonal-corrected synthetic data, the forced-stratified fit
recovers the truth and reports a chi2 equal to its off-diagonal SSR.
"""

from __future__ import annotations

import numpy as np

from tests.optimization.test_phase5_standard_resolver import _laminar_cfg
from xpcsjax.core.diagonal_correction import apply_diagonal_correction
from xpcsjax.core.homodyne_model import HomodyneModel
from xpcsjax.optimization.nlsq import fit_nlsq

_TRUE = np.array([1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0])
_CONTRAST, _OFFSET = 0.3, 1.0


def _fixture(n_phi: int, n_t: int, *, stratified: bool):
    cfg = _laminar_cfg("auto", n_t)
    dt = float(cfg.config["analyzer_parameters"]["dt"])
    t = np.arange(n_t, dtype=np.float64) * dt
    phi = np.linspace(0.0, 90.0, n_phi, dtype=np.float64)
    if stratified:
        # Force the wrapper's full-copy stratified branch on a small dataset
        # (the auto gate needs >= 100k points).
        cfg.config["optimization"]["stratification"] = {
            "enabled": True,
            "target_chunk_size": 200,
        }
    model = HomodyneModel(cfg.config)
    c2 = np.asarray(model.compute_c2(_TRUE, phi, contrast=_CONTRAST, offset=_OFFSET))
    c2 = c2 + np.random.default_rng(3).normal(0.0, 5e-4, size=c2.shape)
    c2 = np.stack([np.asarray(apply_diagonal_correction(c2[i])) for i in range(n_phi)])
    data = {
        "phi_angles_list": phi,
        "c2_exp": c2,
        "t1": t,
        "t2": t,
        "wavevector_q_list": np.array([0.0237]),
    }
    return cfg, model, data


def _off_diagonal_ssr(model, res, data, n_phi, n_t):
    p = np.asarray(res.parameters, dtype=np.float64)
    phys, con, off = p[-7:], p[:n_phi], p[n_phi : 2 * n_phi]
    phi = data["phi_angles_list"]
    mask = ~np.eye(n_t, dtype=bool)
    ssr = 0.0
    for i in range(n_phi):
        k = np.asarray(
            model.compute_c2(phys, phi[i : i + 1], contrast=float(con[i]), offset=float(off[i]))
        )[0]
        ssr += float(np.sum(((k - data["c2_exp"][i])[mask]) ** 2))
    return phys, ssr


def test_forced_stratified_fit_ignores_diagonal_and_matches_standard():
    n_phi, n_t = 3, 30  # 2700 points: standard vs forced-stratified on the same data
    cfg_s, model, data = _fixture(n_phi, n_t, stratified=False)
    cfg_f, _, _ = _fixture(n_phi, n_t, stratified=True)

    res_std = fit_nlsq(data, cfg_s)
    res_strat = fit_nlsq(data, cfg_f)
    assert res_strat.nlsq_diagnostics is not None

    _, ssr_std = _off_diagonal_ssr(model, res_std, data, n_phi, n_t)
    phys_strat, ssr_strat = _off_diagonal_ssr(model, res_strat, data, n_phi, n_t)

    # Diagonal residuals are exactly zero on the stratified branch, so the
    # reported chi2 IS the off-diagonal SSR (rtol covers the per-point kernel's
    # sub-sampled time integral vs the grid kernel, ~1e-3 per point; an unmasked
    # diagonal would add n_phi * n_t * (contrast)^2 ~ 8, i.e. orders more).
    np.testing.assert_allclose(res_strat.chi_squared, ssr_strat, rtol=1e-2)
    # Both paths land on the same (true) physics: no diagonal-driven distortion.
    assert ssr_strat <= ssr_std * 1.5, (ssr_strat, ssr_std)
    np.testing.assert_allclose(phys_strat[0], _TRUE[0], rtol=0.05)  # D0
    np.testing.assert_allclose(phys_strat[1], _TRUE[1], atol=0.05)  # alpha
