"""Task #14 — APPLES-TO-APPLES averaged fit-parity via the compressed-averaged
residual wrapper.

The earlier ``auto_averaged`` engine-route proof
(``test_engine_heterodyne_fit_parity.py``) drove the engine with
``per_angle_scaling=True`` directly, exposing ``2*n_phi`` independent scaling
params — *more* scaling DOF than production's averaged fit (which optimizes a
single ``[avg_contrast, avg_offset]`` pair, 2 DOF). The engine therefore reached
the true minimum partly by having extra per-angle freedom; the comparison was not
apples-to-apples on the scaling DOF count.

This module closes that: it routes the averaged engine through
:func:`wrap_engine_averaged_residual` so the optimizer varies **only the 2
averaged scalars** (compressed physics-first ``[physics | c_avg | o_avg]``,
length ``n_physics + 2``), broadcasting them to the engine's ``2*n_phi``
scaling-first layout *inside* the JIT-traced residual. Both the engine route and
production ``fit_nlsq_multi_phi`` averaged now have exactly the same scaling DOF.

It reuses the well-posed fixture and helpers from the sibling fit-parity module
(same model x0, same NLSQ solver settings, escapes off) and asserts:

1. **DOF compression is real** — the optimized engine param vector length is
   exactly ``n_physics + 2`` (proves the wrapper constrained the problem to 2
   scaling DOF, not ``2*n_phi``).
2. **Wrapped residual fidelity** — at the shared x0 the wrapped (compressed)
   residual is bit-identical to the engine residual at the broadcast scaling-first
   vector (the broadcast is JAX-native and trace-consistent).
3. **No worse than production** — ``chi2_engine <= chi2_ref + tol`` on the
   well-posed surface (a strictly-worse engine objective would be a real bug).
4. **Near-equality with production at matched DOF** — the engine and production
   averaged solves now converge to the SAME averaged minimum (rel_diff ~4e-7).

TASK #14 FINDING (the headline result)
--------------------------------------
The sibling fit-parity module asserted the ``auto_averaged`` engine route reaches
the true global minimum (SSR ~0) while production's averaged solver is "trapped"
at SSR ~1.4 with ``avg_contrast ~0.24`` vs true 0.30. **That was an artifact of
the expanded ``2*n_phi`` scaling DOF**, not a solver-quality difference. With the
engine compressed to the SAME 2 scaling DOF as production (this module), the
engine route lands on the IDENTICAL averaged minimum: ``chi2_engine ~1.4453`` vs
``chi2_ref ~1.4453`` (**rel_diff ~4.2e-7**), recovering the SAME ``avg_contrast
~0.2403``. So the prior "engine is strictly better" was the extra
``2*(n_phi-1)`` per-angle freedom buying a lower objective on a uniform-true-
scaling problem — exactly the apples-to-apples discrepancy three-brain flagged.
At matched DOF the two are equivalent (down to solver-tolerance rel_diff), which
is the correct averaged-mode parity contract.

The ~1.4453 floor is the genuine 2-DOF averaged minimum on this fixture (a single
averaged ``(contrast, offset)`` pair cannot perfectly reproduce the model's true
per-angle structure under soft_l1); production sits at the same floor. This is
NOT ~0 and asserting ``< 1e-6`` here would be wrong — the ~0 was only reachable
with per-angle DOF. We therefore assert no-worse + tight near-equality.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# Reuse the well-posed fixture + harness helpers from the sibling module so the
# x0, solver settings, and frame-0 reconciliation are identical.
from tests.parity.test_engine_heterodyne_fit_parity import (
    _LINUX_ONLY,
    _PER_SET_NFEV,
    _build_engine,
    _drop_frame0_stratified_data,
    _make_well_posed_case,
)
from xpcsjax.config.parameter_registry import SCALING_PARAMS
from xpcsjax.optimization.nlsq.heterodyne_adapter import NLSQAdapter
from xpcsjax.optimization.nlsq.heterodyne_averaged_wrapper import (
    compressed_averaged_to_engine_scaling_first,
    engine_popt_to_compressed_averaged,
    wrap_engine_averaged_residual,
)
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi
from xpcsjax.optimization.nlsq.heterodyne_layout import physics_first_to_scaling_first
from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
    build_heterodyne_stratified_data,
)
from xpcsjax.optimization.nlsq.strategies.heterodyne_hybrid_streaming import (
    build_heterodyne_pointwise_model,
)
from xpcsjax.optimization.nlsq.strategies.stratified_ls import create_stratified_chunks


def _run_averaged_wrapper_case():
    """Run production averaged (2 DOF) and the wrapper-routed engine averaged (2
    DOF) on the SAME well-posed fixture, returning objectives + diagnostics."""
    model, c2, phi = _make_well_posed_case()
    phys_names = list(model.param_manager.varying_names)
    n_varying = len(phys_names)
    n_phi = len(phi)
    t = np.asarray(model.t, dtype=np.float64)
    q, dt = float(model.q), float(model.dt)

    # ---- Reference: production averaged joint fit (auto -> averaged at n_phi=6) ----
    ref_cfg = NLSQConfig(
        method="trf",
        loss="soft_l1",
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        x_scale="jac",
        max_nfev=_PER_SET_NFEV,
        enable_cmaes=False,
        multistart=False,
    )
    ref_cfg.per_angle_mode = "auto"
    result_ref = fit_nlsq_multi_phi(model, c2, list(phi), ref_cfg, None)
    chi2_ref = float(result_ref.chi_squared)

    # ---- Engine route: COMPRESSED averaged (2 scaling DOF) via the wrapper ----
    strat_full = build_heterodyne_stratified_data(model, c2, np.asarray(phi))
    strat = _drop_frame0_stratified_data(strat_full, t=t, n_phi=n_phi)
    chunked = create_stratified_chunks(strat, target_chunk_size=100_000)

    _mf, _x, _y, p0, meta = build_heterodyne_pointwise_model(
        stratified_data=strat_full,
        model=model,
        physical_param_names=phys_names,
        per_angle_mode="auto_averaged",
    )
    # p0 is the COMPRESSED physics-first averaged vector [physics | c_avg | o_avg].
    p0 = np.asarray(p0, dtype=np.float64)
    assert p0.shape == (n_varying + 2,), (
        f"compressed averaged p0 length {p0.shape} != ({n_varying + 2},)"
    )

    contrast_arr = np.asarray(meta["contrast_arr"], dtype=np.float64)
    offset_arr = np.asarray(meta["offset_arr"], dtype=np.float64)

    # Build the per-angle-scaling engine (the wrapper feeds it the broadcast vec).
    engine = _build_engine(
        mode="auto_averaged",
        chunked=chunked,
        phys_names=phys_names,
        contrast_arr=contrast_arr,
        offset_arr=offset_arr,
        q=q,
        dt=dt,
    )

    # COMPRESSED bounds: physics bounds + a single (contrast, offset) pair.
    physics_lower, physics_upper = model.param_manager.get_bounds()
    cb = (SCALING_PARAMS["contrast"].min_bound, SCALING_PARAMS["contrast"].max_bound)
    ob = (SCALING_PARAMS["offset"].min_bound, SCALING_PARAMS["offset"].max_bound)
    lb_c = np.concatenate([np.asarray(physics_lower, dtype=np.float64), [cb[0], ob[0]]])
    ub_c = np.concatenate([np.asarray(physics_upper, dtype=np.float64), [cb[1], ob[1]]])

    # The wrapper makes the optimizer vary ONLY the compressed (n_physics + 2) vector.
    wrapped = wrap_engine_averaged_residual(lambda v: engine(v), n_physics=n_varying, n_phi=n_phi)

    # Fidelity check (investigation-first): wrapped residual at p0 == engine
    # residual at the broadcast scaling-first vector built by the numpy helper.
    sf_p0 = physics_first_to_scaling_first(
        p0, n_physics=n_varying, mode="auto_averaged", n_phi=n_phi
    )
    r_wrapped = np.asarray(wrapped(jnp.asarray(p0)), dtype=np.float64)
    r_engine = np.asarray(engine(jnp.asarray(sf_p0)), dtype=np.float64)
    max_resid_diff = float(np.max(np.abs(r_wrapped - r_engine)))

    def residual_fn(x: np.ndarray):
        return wrapped(jnp.asarray(x, dtype=jnp.float64))

    joint_cfg = NLSQConfig(
        method="trf",
        loss="soft_l1",
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        x_scale="jac",
        max_nfev=_PER_SET_NFEV * n_phi,
        n_params=len(p0),
    )
    adapter = NLSQAdapter(parameter_names=[f"p{i}" for i in range(len(p0))])
    res = adapter.fit(
        residual_fn=residual_fn,
        initial_params=p0,
        bounds=(lb_c, ub_c),
        config=joint_cfg,
    )
    popt_compressed = np.asarray(res.parameters, dtype=np.float64)
    resid_at_opt = np.asarray(wrapped(jnp.asarray(popt_compressed)), dtype=np.float64)
    chi2_engine = float(np.sum(resid_at_opt**2))

    # popt boundary passthrough (already physics-first compressed).
    popt_pf = engine_popt_to_compressed_averaged(popt_compressed, n_physics=n_varying)

    return {
        "chi2_ref": chi2_ref,
        "chi2_engine": chi2_engine,
        "n_varying": n_varying,
        "opt_len": len(popt_compressed),
        "max_resid_diff": max_resid_diff,
        "engine_success": bool(res.success),
        "popt_pf": popt_pf,
    }


def test_averaged_wrapper_residual_is_jax_native_and_trace_consistent():
    """The wrapper broadcast is JAX-native: at a fixed compressed x the wrapped
    residual equals the engine residual at the broadcast scaling-first vector
    (bit-identical), and the broadcast maps the 2 scalars to 2*n_phi uniformly."""
    n_physics, n_phi = 14, 6
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n_physics + 2)
    sf = np.asarray(
        compressed_averaged_to_engine_scaling_first(
            jnp.asarray(x), n_physics=n_physics, n_phi=n_phi
        )
    )
    assert sf.shape == (2 * n_phi + n_physics,)
    # contrast block uniform == c_avg; offset block uniform == o_avg; physics intact.
    assert np.allclose(sf[:n_phi], x[n_physics])
    assert np.allclose(sf[n_phi : 2 * n_phi], x[n_physics + 1])
    assert np.allclose(sf[2 * n_phi :], x[:n_physics])
    # Matches the existing numpy boundary helper exactly.
    sf_numpy = physics_first_to_scaling_first(
        x, n_physics=n_physics, mode="auto_averaged", n_phi=n_phi
    )
    assert np.array_equal(sf, sf_numpy)


@_LINUX_ONLY
def test_averaged_wrapper_apples_to_apples_two_dof():
    """APPLES-TO-APPLES averaged parity: engine route at exactly 2 scaling DOF
    (via the wrapper) vs production averaged (also 2 DOF), same well-posed fixture.

    KEY checks:
    * the optimized engine param-vector length == n_physics + 2 (the compression
      actually constrains it to 2 scaling DOF — not 2*n_phi);
    * the wrapped residual is trace-consistent with the engine (max abs diff 0);
    * engine objective is no worse than production (chi2_engine <= chi2_ref + tol);
    * at matched 2 DOF the engine converges to the SAME averaged minimum as
      production (rel_diff ~4e-7) — the prior "engine reaches ~0" was an artifact
      of the expanded 2*n_phi DOF, not solver quality (see module docstring).
    """
    out = _run_averaged_wrapper_case()
    chi2_ref = out["chi2_ref"]
    chi2_engine = out["chi2_engine"]
    n_varying = out["n_varying"]

    # (1) DOF compression is real: optimizer saw exactly n_physics + 2 params.
    assert out["opt_len"] == n_varying + 2, (
        f"averaged engine route optimized {out['opt_len']} params; expected "
        f"n_physics+2={n_varying + 2} (2 scaling DOF). The wrapper failed to "
        "compress the scaling block — it would otherwise expose 2*n_phi DOF."
    )

    # (2) Wrapper residual fidelity: bit-identical to the engine at the broadcast.
    assert out["max_resid_diff"] == 0.0, (
        f"wrapped residual differs from engine-at-broadcast by "
        f"{out['max_resid_diff']!r}; the JAX broadcast is not trace-consistent."
    )

    assert np.isfinite(chi2_ref) and np.isfinite(chi2_engine)
    rel = abs(chi2_engine - chi2_ref) / max(abs(chi2_ref), 1e-300)

    # (3) No worse than production on a well-posed surface.
    assert chi2_engine <= chi2_ref * (1.0 + 1e-3) + 1e-12, (
        f"averaged engine objective {chi2_engine!r} is STRICTLY WORSE than "
        f"production {chi2_ref!r} (rel_diff={rel:.3e}) at matched 2 scaling DOF — "
        "a real residual/scaling/layout/solver bug. Do NOT loosen this."
    )

    # (4) Near-equality with production at MATCHED 2 scaling DOF. This is the
    # Task #14 headline: the sibling module's "engine reaches ~0, production
    # trapped at ~1.4" gap was an artifact of the engine's expanded 2*n_phi DOF.
    # Compressed to the same 2 DOF, the engine converges to the SAME averaged
    # minimum (~1.4453) production does (rel_diff ~4.2e-7). A LARGE divergence
    # here (engine still reaching ~0, or wandering far from production) at matched
    # DOF would itself be a real finding — diagnose it, do NOT loosen this.
    assert rel < 1e-4, (
        f"averaged engine objective {chi2_engine!r} diverges from production "
        f"{chi2_ref!r} (rel_diff={rel:.3e}) at MATCHED 2 scaling DOF. At equal DOF "
        "the two averaged solves should converge to the same minimum; a large "
        "rel_diff means the compression/broadcast is not apples-to-apples — a real "
        "Task #14 finding (residual/scaling/layout/solver bug)."
    )
