"""Task #15 — Real-data engine-vs-production fit-parity validation on C044.

GATING QUESTION
---------------
A noiseless-synthetic fit-parity finding (``tests/parity/test_engine_heterodyne_fit_parity.py``)
showed that routing ``two_component`` through the shared homodyne stratification
engine (``StratifiedResidualFunctionJIT``) gives a *better-conditioned* residual
than the production in-memory joint fit (``fit_nlsq_multi_phi``): for
``individual`` mode the engine reaches the true minimum where production stays
trapped (even with escapes on), while ``fixed_constant`` is strict-parity.

That finding lives on a NOISELESS fixture where the true minimum is known
(SSR ~0). This script answers the deliverable question:

    Does ``chi2_engine <= chi2_production`` (no-worse) hold on REAL, NOISY C044
    data where the true minimum is UNKNOWN — or was the advantage a
    noiseless-fixture artifact?

METHOD (mirrors tests/parity/test_engine_heterodyne_fit_parity.py)
------------------------------------------------------------------
The engine-route construction, layout conversion, frame-0 exclusion, and the
production-fit driving are LIFTED from that harness's ``_run_reference_and_engine``.
The ONLY substitution is the data source: instead of the synthetic well-posed
fixture (``_make_well_posed_case``), we load REAL C044 two-time data via
``load_xpcs_data`` and crop the two-time matrix to a small time window so the
in-memory joint fit (the path the finding is about) is tractable in seconds. The
data stays REAL and NOISY — we only keep fewer time lags.

Modes tested: ``fixed_constant`` and ``individual`` (NOT ``auto_averaged`` — its
compressed-averaged engine path isn't built, so it isn't apples-to-apples).

Both fits start from the SAME x0 (config-initial params) and use the SAME NLSQ
solver settings (trf / soft_l1 / ftol=xtol=gtol=1e-8 / x_scale=jac), CMA-ES and
multistart OFF (plain in-memory joint fit). We report ``chi2_engine`` vs
``chi2_ref``, rel_diff, and the no-worse verdict.

This script touches NO production code; it only READS the production dispatch and
the engine modules, exactly as the test harness does.

Run:
    uv run python scripts/realdata_engine_fit_parity_c044.py \
        [--config /path/to/xpcsjax_config.yaml] [--n-t 64] [--n-phi 0] [--nfev 600]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import xpcsjax  # noqa: F401  -- sets JAX_ENABLE_X64 before any jax import
from xpcsjax.config import ConfigManager
from xpcsjax.config.parameter_registry import SCALING_PARAMS
from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
from xpcsjax.data import load_xpcs_data
from xpcsjax.optimization.nlsq.heterodyne_adapter import NLSQAdapter
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi
from xpcsjax.optimization.nlsq.heterodyne_layout import (
    physics_first_to_scaling_first,
    scaling_first_to_physics_first,
)
from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
    HeterodyneStratifiedData,
    build_heterodyne_stratified_data,
)
from xpcsjax.optimization.nlsq.model_adapter import HeterodynePointEvaluator
from xpcsjax.optimization.nlsq.strategies.heterodyne_hybrid_streaming import (
    build_heterodyne_pointwise_model,
)
from xpcsjax.optimization.nlsq.strategies.residual_jit import (
    StratifiedResidualFunctionJIT,
)
from xpcsjax.optimization.nlsq.strategies.stratified_ls import (
    create_stratified_chunks,
)

_DEFAULT_CONFIG = "/home/wei/Documents/Projects/data/C044/xpcsjax_config.yaml"
_MODES = ("fixed_constant", "individual")
_MODE_TO_PRODUCTION = {
    "fixed_constant": "constant",
    "individual": "individual",
}


# ---------------------------------------------------------------------------
# Engine construction + frame-0 exclusion — copied verbatim (logic) from the
# test harness, so the engine route is built identically.
# ---------------------------------------------------------------------------
def _drop_frame0_stratified_data(
    strat: HeterodyneStratifiedData, *, t: np.ndarray, n_phi: int
) -> HeterodyneStratifiedData:
    t = np.asarray(t, dtype=np.float64)
    t0 = float(t[0])
    eps = float(strat.dt) * 1e-6
    keep = (strat.t1_flat > t0 + eps) & (strat.t2_flat > t0 + eps)

    new_sizes: list[int] = []
    cursor = 0
    for size in strat.chunk_sizes:
        new_sizes.append(int(np.sum(keep[cursor : cursor + size])))
        cursor += size

    n_t_reduced = len(t) - 1
    return HeterodyneStratifiedData(
        phi_flat=strat.phi_flat[keep].copy(),
        t1_flat=strat.t1_flat[keep].copy(),
        t2_flat=strat.t2_flat[keep].copy(),
        g2_flat=strat.g2_flat[keep].copy(),
        sigma=np.ones((n_phi, n_t_reduced, n_t_reduced), dtype=np.float64),
        q=strat.q,
        L=strat.L,
        dt=strat.dt,
        chunk_sizes=new_sizes,
        n_phi=strat.n_phi,
        n_t=n_t_reduced,
        angle_indices=list(strat.angle_indices),
    )


def _build_engine(
    *,
    mode: str,
    chunked,
    phys_names: list[str],
    contrast_arr: np.ndarray,
    offset_arr: np.ndarray,
    q: float,
    dt: float,
) -> StratifiedResidualFunctionJIT:
    evaluator = HeterodynePointEvaluator(
        analysis_mode="two_component", q=float(q), dt=float(dt)
    )
    if mode == "fixed_constant":
        return StratifiedResidualFunctionJIT(
            stratified_data=chunked,
            per_angle_scaling=False,
            physical_param_names=phys_names,
            fixed_contrast_per_angle=np.asarray(contrast_arr, dtype=np.float64),
            fixed_offset_per_angle=np.asarray(offset_arr, dtype=np.float64),
            evaluator=evaluator,
        )
    return StratifiedResidualFunctionJIT(
        stratified_data=chunked,
        per_angle_scaling=True,
        physical_param_names=phys_names,
        fixed_contrast_per_angle=None,
        fixed_offset_per_angle=None,
        evaluator=evaluator,
    )


def _physics_first_bounds(
    *, mode: str, n_phi: int, physics_lower: np.ndarray, physics_upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    cb = (SCALING_PARAMS["contrast"].min_bound, SCALING_PARAMS["contrast"].max_bound)
    ob = (SCALING_PARAMS["offset"].min_bound, SCALING_PARAMS["offset"].max_bound)
    physics_lower = np.asarray(physics_lower, dtype=np.float64)
    physics_upper = np.asarray(physics_upper, dtype=np.float64)

    if mode == "fixed_constant":
        return physics_lower, physics_upper
    # individual: tail [contrast(n_phi) | offset(n_phi)]
    lb = np.concatenate([physics_lower, np.full(n_phi, cb[0]), np.full(n_phi, ob[0])])
    ub = np.concatenate([physics_upper, np.full(n_phi, cb[1]), np.full(n_phi, ob[1])])
    return lb, ub


# ---------------------------------------------------------------------------
# Real-data loading + time crop (the ONLY substitution vs the test harness).
# ---------------------------------------------------------------------------
def load_real_subset(config_path: str, *, n_t: int, n_phi: int):
    """Load REAL C044 data, crop to a small time window (and optionally a subset
    of angles). Returns (model, c2_sub, phi_sub, info).

    The crop keeps the first ``n_t`` time indices of the two-time matrix — this is
    still REAL noisy data, just fewer time lags so the in-memory joint fit is
    tractable. ``n_phi=0`` keeps ALL angles.

    The C044 config uses relative data paths (``./``) resolved against the CWD, so
    we temporarily chdir into the config's directory for the load.
    """
    import os
    from pathlib import Path

    cfg_dir = str(Path(config_path).resolve().parent)
    prev_cwd = os.getcwd()
    try:
        os.chdir(cfg_dir)
        data = load_xpcs_data(config_path)
    finally:
        os.chdir(prev_cwd)

    def _get(d, keys):
        for k in keys:
            if k in d:
                return d[k]
        raise KeyError(f"none of {keys} in data keys {list(d.keys())}")

    c2_full = np.asarray(_get(data, ("c2_exp", "c2")), dtype=np.float64)
    phi_full = np.asarray(_get(data, ("phi_angles_list", "phi_angles", "phi")), dtype=np.float64)

    n_phi_full, nt1, nt2 = c2_full.shape
    assert nt1 == nt2, f"non-square two-time matrix: {c2_full.shape}"

    keep_t = min(int(n_t), nt1)
    if n_phi and int(n_phi) < n_phi_full:
        # evenly-spaced angle subset for broad phi coverage
        idx = np.linspace(0, n_phi_full - 1, int(n_phi)).round().astype(int)
        idx = np.unique(idx)
        c2_sub = c2_full[idx][:, :keep_t, :keep_t].copy()
        phi_sub = phi_full[idx].copy()
    else:
        c2_sub = c2_full[:, :keep_t, :keep_t].copy()
        phi_sub = phi_full.copy()

    # Build the model with a time grid that matches the CROPPED two-time matrix.
    # We keep the FIRST ``keep_t`` time indices of c2, so the model's native grid
    # must also be the first ``keep_t`` frames: set end_frame = start_frame+keep_t-1.
    # (Rebuilding from config rather than calling sync_time_axis, which trims from
    # the trailing edge — the wrong end for a leading-window crop.)
    cfg = ConfigManager(config_path)
    assert cfg.config is not None
    cfg_dict = dict(cfg.config)
    ap = dict(cfg_dict.get("analyzer_parameters", {}))
    if "start_frame" in ap and "end_frame" in ap:
        start_frame = int(ap["start_frame"])
        ap["end_frame"] = start_frame + keep_t - 1
        cfg_dict["analyzer_parameters"] = ap
    model = HeterodyneModel.from_config(cfg_dict)
    assert int(model.n_times) == keep_t, (
        f"model n_times {int(model.n_times)} != cropped n_t {keep_t}"
    )

    info = {
        "config_path": config_path,
        "c2_full_shape": list(c2_full.shape),
        "n_phi_used": int(c2_sub.shape[0]),
        "n_t_used": int(keep_t),
        "total_points_used": int(c2_sub.size),
        "phi_used": [float(x) for x in phi_sub],
        "phi_full_count": int(n_phi_full),
    }
    return model, c2_sub, phi_sub, info


# ---------------------------------------------------------------------------
# Run production reference + engine route for one mode (mirrors the harness).
# ---------------------------------------------------------------------------
def run_reference_and_engine(model, c2, phi, *, mode: str, nfev: int) -> dict:
    import jax.numpy as jnp

    phys_names = list(model.param_manager.varying_names)
    n_varying = len(phys_names)
    n_phi = len(phi)
    # IMPORTANT: the model's own time grid must match the cropped two-time matrix
    # the engine/strat builder reads. We crop the model's t to n_t as well.
    t_full = np.asarray(model.t, dtype=np.float64)
    n_t = c2.shape[1]
    t = t_full[:n_t]
    q, dt = float(model.q), float(model.dt)

    # ---- Reference: production heterodyne joint fit (CMA-ES / multistart OFF) ----
    ref_cfg = NLSQConfig(
        method="trf",
        loss="soft_l1",
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        x_scale="jac",
        max_nfev=nfev,
        enable_cmaes=False,
        multistart=False,
    )
    ref_cfg.per_angle_mode = _MODE_TO_PRODUCTION[mode]
    t_ref0 = time.time()
    result_ref = fit_nlsq_multi_phi(model, c2, list(phi), ref_cfg, None)
    ref_wall = time.time() - t_ref0
    chi2_ref = float(result_ref.chi_squared)
    diag = result_ref.nlsq_diagnostics or {}
    popt_ref = np.asarray(result_ref.parameters, dtype=np.float64)
    physics_ref = popt_ref[:n_varying]

    # ---- Engine route: same NLSQAdapter solver over the engine residual ----
    strat_full = build_heterodyne_stratified_data(model, c2, np.asarray(phi))
    strat = _drop_frame0_stratified_data(strat_full, t=t, n_phi=n_phi)
    chunked = create_stratified_chunks(strat, target_chunk_size=100_000)

    _mf, _x, _y, p0, meta = build_heterodyne_pointwise_model(
        stratified_data=strat_full,
        model=model,
        physical_param_names=phys_names,
        per_angle_mode=mode,
    )
    p0 = np.asarray(p0, dtype=np.float64)

    # Frozen scaling for fixed_constant: production's quantile estimate
    # (sorted-phi order) so engine + reference freeze the SAME constants.
    if mode == "fixed_constant":
        contrast_arr = np.asarray(diag["contrast_per_angle_fixed"], dtype=np.float64)
        offset_arr = np.asarray(diag["offset_per_angle_fixed"], dtype=np.float64)
    else:
        contrast_arr = np.asarray(meta["contrast_arr"], dtype=np.float64)
        offset_arr = np.asarray(meta["offset_arr"], dtype=np.float64)

    physics_lower, physics_upper = model.param_manager.get_bounds()
    lb_pf, ub_pf = _physics_first_bounds(
        mode=mode, n_phi=n_phi, physics_lower=physics_lower, physics_upper=physics_upper
    )

    x0_sf = physics_first_to_scaling_first(p0, n_physics=n_varying, mode=mode, n_phi=n_phi)
    lb_sf = physics_first_to_scaling_first(lb_pf, n_physics=n_varying, mode=mode, n_phi=n_phi)
    ub_sf = physics_first_to_scaling_first(ub_pf, n_physics=n_varying, mode=mode, n_phi=n_phi)

    expected_len = n_varying if mode == "fixed_constant" else 2 * n_phi + n_varying
    assert x0_sf.shape == (expected_len,), (
        f"mode={mode}: engine x0 length {x0_sf.shape} != ({expected_len},)"
    )

    engine = _build_engine(
        mode=mode,
        chunked=chunked,
        phys_names=phys_names,
        contrast_arr=contrast_arr,
        offset_arr=offset_arr,
        q=q,
        dt=dt,
    )

    def residual_fn(x: np.ndarray):
        return engine(jnp.asarray(x, dtype=jnp.float64))

    joint_cfg = NLSQConfig(
        method="trf",
        loss="soft_l1",
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        x_scale="jac",
        max_nfev=nfev * n_phi,
        n_params=len(x0_sf),
    )
    adapter = NLSQAdapter(parameter_names=[f"p{i}" for i in range(len(x0_sf))])
    t_eng0 = time.time()
    res = adapter.fit(
        residual_fn=residual_fn,
        initial_params=x0_sf,
        bounds=(lb_sf, ub_sf),
        config=joint_cfg,
    )
    eng_wall = time.time() - t_eng0
    popt_sf = np.asarray(res.parameters, dtype=np.float64)
    resid_at_opt = np.asarray(engine(jnp.asarray(popt_sf)), dtype=np.float64)
    chi2_engine = float(np.sum(resid_at_opt**2))

    # Also evaluate the engine residual at x0 (shared start) as a sanity anchor.
    resid_at_x0 = np.asarray(engine(jnp.asarray(x0_sf)), dtype=np.float64)
    chi2_engine_x0 = float(np.sum(resid_at_x0**2))

    popt_pf = scaling_first_to_physics_first(
        popt_sf, n_physics=n_varying, mode=mode, n_phi=n_phi
    )

    rel_diff = (chi2_engine - chi2_ref) / max(abs(chi2_ref), 1e-300)
    # No-worse tolerance: 1e-3 relative (same as the harness's keep-better gate).
    no_worse = chi2_engine <= chi2_ref * (1.0 + 1e-3)

    return {
        "mode": mode,
        "production_per_angle_mode": _MODE_TO_PRODUCTION[mode],
        "n_engine_residual_support": int(resid_at_opt.size),
        "chi2_ref": chi2_ref,
        "chi2_engine": chi2_engine,
        "chi2_engine_at_x0": chi2_engine_x0,
        "rel_diff": float(rel_diff),
        "no_worse": bool(no_worse),
        "ref_convergence_status": str(result_ref.convergence_status),
        "engine_success": bool(res.success),
        "ref_wall_s": round(ref_wall, 2),
        "engine_wall_s": round(eng_wall, 2),
        "physics_ref": [float(x) for x in physics_ref],
        "physics_engine": [float(x) for x in popt_pf[:n_varying]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=_DEFAULT_CONFIG)
    ap.add_argument("--n-t", type=int, default=64, help="time-window crop (first n_t indices)")
    ap.add_argument("--n-phi", type=int, default=0, help="0 = all angles, else subset count")
    ap.add_argument("--nfev", type=int, default=600, help="per-angle max_nfev")
    args = ap.parse_args()

    print(f"[load] config={args.config}  n_t={args.n_t}  n_phi={args.n_phi}")
    model, c2, phi, info = load_real_subset(args.config, n_t=args.n_t, n_phi=args.n_phi)
    print("[load] subset:", json.dumps(info, indent=2))

    results = []
    for mode in _MODES:
        print(f"\n[run] mode={mode} ...")
        out = run_reference_and_engine(model, c2, phi, mode=mode, nfev=args.nfev)
        results.append(out)
        print(
            f"[run] mode={mode}: chi2_ref={out['chi2_ref']:.6e}  "
            f"chi2_engine={out['chi2_engine']:.6e}  rel_diff={out['rel_diff']:+.3e}  "
            f"no_worse={out['no_worse']}  ref_status={out['ref_convergence_status']}"
        )

    print("\n" + "=" * 78)
    print("REAL-DATA C044 ENGINE-vs-PRODUCTION FIT PARITY — SUMMARY")
    print("=" * 78)
    print(json.dumps({"subset": info, "results": results}, indent=2))

    print("\nVERDICT TABLE")
    print(f"{'mode':<16}{'chi2_ref':>16}{'chi2_engine':>16}{'rel_diff':>14}{'no_worse':>10}")
    for r in results:
        print(
            f"{r['mode']:<16}{r['chi2_ref']:>16.6e}{r['chi2_engine']:>16.6e}"
            f"{r['rel_diff']:>+14.3e}{str(r['no_worse']):>10}"
        )

    all_no_worse = all(r["no_worse"] for r in results)
    print(f"\nBOTTOM LINE: engine no-worse on ALL tested modes = {all_no_worse}")


if __name__ == "__main__":
    main()
