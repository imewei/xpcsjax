"""Engine-route ``two_component`` fit + heterodyne result contract (Tasks #16a/#16b).

Runs the heterodyne (``two_component``) joint fit through the **shared homodyne
stratification engine** (:class:`StratifiedResidualFunctionJIT`) and returns a
complete heterodyne-contract :class:`OptimizationResult` — the same keys, shapes
and conventions production callers (``fit_nlsq_multi_phi`` →
``_fit_joint_constant_multi_phi`` / ``_fit_joint_averaged_multi_phi`` /
``_fit_joint_multi_phi``) emit.

**Wired into production (Task #16b).** ``_fit_nlsq_heterodyne`` routes the
in-memory joint fit (< 1 M points, non-escape) for the three in-scope per-angle
scaling modes through :func:`fit_two_component_via_engine` here, best-effort with
a fall-back to ``fit_nlsq_multi_phi`` on any engine-route exception. This
**changes** ``two_component`` in-memory in-scope-mode results by ~1e-3 vs the old
direct path under the accepted *no-worse* contract (engine SSR ≤ production SSR),
**not** bit-identical — see ``CLAUDE.md``. The superseded seed-42 angle-shuffle
regime was removed alongside the flip; the off-grid guard is obsolete on this
path because :class:`HeterodynePointEvaluator` uses the meshgrid kernel (no
value→index mapping). The proven engine construction (frame-0-excluded chunks +
per-mode :class:`HeterodynePointEvaluator` + physics-first⇄scaling-first layout)
was promoted here from the Phase 2.3 parity tests
(``tests/parity/test_engine_heterodyne_fit_parity.py``) so the production path no
longer imports engine-construction helpers from tests.

Scope: the three in-scope per-angle scaling modes —
``fixed_constant`` / ``individual`` / ``auto_averaged`` (the engine-layout
tokens for production ``constant`` / ``individual`` / ``auto``-at-``n_phi>=3``).
``fourier`` is out of scope here and raises :class:`NotImplementedError`
(#16b keeps it on the existing path).

Result-contract assembly REUSES the production result-builder primitives:

* ``compute_multi_angle_residuals`` — the production residual on the production
  support (``(n_t-1)*(n_t-2)`` per angle), so ``chi_squared`` / ``chi2_per_angle``
  are byte-for-byte the production objective, not the engine's own SSR.
* ``_decompose_chi2_per_angle`` — per-angle χ² decomposition (SSR conservation).
* ``noise_normalized_reduced_chi2`` — the shared reduced-χ² correction.
* ``_build_heterodyne_diagnostics`` — the symmetric ``nlsq_diagnostics`` dict
  (``per_angle_mode`` / ``chi2_per_angle`` / ``scaling_source`` /
  ``fourier_basis_dim`` + the anti-degeneracy activation block).

so each mode's returned ``OptimizationResult`` matches the contract of the
production path it mirrors.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from xpcsjax.config.parameter_registry import SCALING_PARAMS
from xpcsjax.optimization.nlsq.heterodyne_averaged_wrapper import (
    engine_popt_to_compressed_averaged,
    wrap_engine_averaged_residual,
)
from xpcsjax.optimization.nlsq.heterodyne_layout import (
    IN_SCOPE_MODES,
    physics_first_to_scaling_first,
    scaling_first_to_physics_first,
)

if TYPE_CHECKING:
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
        HeterodyneStratifiedData,
    )
    from xpcsjax.optimization.nlsq.results import OptimizationResult
    from xpcsjax.optimization.nlsq.strategies.residual_jit import (
        StratifiedResidualFunctionJIT,
    )

__all__ = ["fit_two_component_via_engine", "PRODUCTION_TO_ENGINE_MODE"]

# Production ``_resolve_effective_mode`` token -> engine-layout mode token. The
# layout module (``heterodyne_layout.IN_SCOPE_MODES``) names the engine side
# ``fixed_constant`` / ``auto_averaged`` / ``individual``; the production
# resolver returns ``constant`` / ``averaged`` / ``individual`` / ``fourier``.
PRODUCTION_TO_ENGINE_MODE: dict[str, str] = {
    "constant": "fixed_constant",
    "averaged": "auto_averaged",
    "individual": "individual",
}


# ---------------------------------------------------------------------------
# Engine construction (promoted from tests/parity/test_engine_heterodyne_fit_parity.py)
# ---------------------------------------------------------------------------
def _drop_frame0_stratified_data(
    strat: HeterodyneStratifiedData, *, t: np.ndarray, n_phi: int
) -> HeterodyneStratifiedData:
    """Drop every ``(t1, t2)`` pair touching frame-0, keeping the on-grid diagonal.

    The engine masks the diagonal itself; after that masking the contributing
    support is ``(n_t-1)*(n_t-2)`` per angle — EXACTLY the production support
    (``compute_multi_angle_residuals`` excludes both the diagonal and the t=0
    boundary). Removing frame-0 from the flat data is physics-safe under
    ``JAX_ENABLE_X64=1``: the heterodyne transport/velocity terms are cumsum
    DIFFERENCES, so shifting the cumsum anchor by a constant cancels in every
    interior ``(i>0, j>0)`` pair (verified to ~4e-15 rel in Phase 2.3a).
    """
    from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
        HeterodyneStratifiedData,
    )

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
    chunked: Any,
    phys_names: list[str],
    contrast_arr: np.ndarray,
    offset_arr: np.ndarray,
    q: float,
    dt: float,
) -> StratifiedResidualFunctionJIT:
    """Construct the frame-0-excluded engine for ``mode`` (mirrors Phase 2.3a).

    ``fixed_constant`` freezes the per-angle scaling on the engine; the other two
    modes expose the engine's ``2*n_phi`` per-angle scaling-first layout (the
    ``auto_averaged`` 2→2*n_phi broadcast happens at the optimizer boundary via
    :func:`physics_first_to_scaling_first`).
    """
    from xpcsjax.optimization.nlsq.model_adapter import HeterodynePointEvaluator
    from xpcsjax.optimization.nlsq.strategies.residual_jit import (
        StratifiedResidualFunctionJIT,
    )

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
    """Physics-first ``(lb, ub)`` matching the production joint-fit layout."""
    cb = (SCALING_PARAMS["contrast"].min_bound, SCALING_PARAMS["contrast"].max_bound)
    ob = (SCALING_PARAMS["offset"].min_bound, SCALING_PARAMS["offset"].max_bound)
    physics_lower = np.asarray(physics_lower, dtype=np.float64)
    physics_upper = np.asarray(physics_upper, dtype=np.float64)

    if mode == "fixed_constant":
        return physics_lower, physics_upper
    if mode == "auto_averaged":
        lb = np.concatenate([physics_lower, [cb[0], ob[0]]])
        ub = np.concatenate([physics_upper, [cb[1], ob[1]]])
        return lb, ub
    # individual: tail [contrast(n_phi) | offset(n_phi)]
    lb = np.concatenate([physics_lower, np.full(n_phi, cb[0]), np.full(n_phi, ob[0])])
    ub = np.concatenate([physics_upper, np.full(n_phi, cb[1]), np.full(n_phi, ob[1])])
    return lb, ub


def _quantile_frozen_scaling(
    model: HeterodyneModel, c2_data: np.ndarray, n_phi: int
) -> tuple[np.ndarray, np.ndarray]:
    """Production constant-mode frozen ``(contrast, offset)`` per angle.

    Mirrors ``_fit_joint_constant_multi_phi`` EXACTLY: flatten the full grid,
    run :func:`estimate_per_angle_scaling_from_quantile` at quantile 0.95, clamp
    to the registry bounds. Sourcing the engine's frozen scaling from the SAME
    estimator the production constant path uses is what makes ``fixed_constant``
    reach the identical physics-only minimum (Phase 2.3b STEP-0 finding).
    """
    from xpcsjax.core.heterodyne_scaling_utils import (
        estimate_per_angle_scaling_from_quantile,
    )
    from xpcsjax.optimization.nlsq.heterodyne_constant_mode import _flatten_inputs

    c2_flat, t1_flat, t2_flat, phi_idx_flat = _flatten_inputs(model, c2_data, n_phi)
    contrast_fixed, offset_fixed = estimate_per_angle_scaling_from_quantile(
        c2_data=c2_flat,
        t1=t1_flat,
        t2=t2_flat,
        phi_indices=phi_idx_flat,
        n_phi=n_phi,
        quantile=0.95,
    )
    contrast_info = SCALING_PARAMS["contrast"]
    offset_info = SCALING_PARAMS["offset"]
    contrast_fixed = np.clip(
        contrast_fixed, contrast_info.min_bound, contrast_info.max_bound
    ).astype(np.float64)
    offset_fixed = np.clip(
        offset_fixed, offset_info.min_bound, offset_info.max_bound
    ).astype(np.float64)
    return contrast_fixed, offset_fixed


# ---------------------------------------------------------------------------
# Production-support objective (REUSE the production residual + helpers)
# ---------------------------------------------------------------------------
def _production_support_chi2(
    *,
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: np.ndarray,
    physics_varying: np.ndarray,
    contrast_per_angle: np.ndarray,
    offset_per_angle: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[float, np.ndarray]:
    """``(chi_squared, chi2_per_angle)`` on the PRODUCTION support.

    Evaluates the production residual ``compute_multi_angle_residuals`` (which
    excludes the diagonal AND the t=0 boundary → ``(n_t-1)*(n_t-2)`` per angle)
    at the engine-fitted physics + per-angle scaling, then decomposes per angle
    via the production helper. Using the production residual (not the engine's
    own SSR) guarantees ``chi2_per_angle.sum() == chi_squared`` on the same
    objective the production paths report — the heterodyne contract invariant.
    """
    import jax.numpy as jnp

    from xpcsjax.core.heterodyne_jax_backend import compute_multi_angle_residuals
    from xpcsjax.optimization.nlsq.heterodyne_constant_mode import (
        _decompose_chi2_per_angle,
    )

    n_phi = len(phi_angles)
    t = np.asarray(model.t, dtype=np.float64)
    full = np.asarray(model.param_manager.get_full_values(), dtype=np.float64).copy()
    full[np.asarray(model.param_manager.varying_indices)] = np.asarray(
        physics_varying, dtype=np.float64
    )

    c2_batch = jnp.asarray(c2_data, dtype=jnp.float64)
    if weights is None:
        weights_batch = jnp.ones_like(c2_batch)
    else:
        w = jnp.asarray(weights, dtype=jnp.float64)
        if w.ndim == 2:
            w = jnp.broadcast_to(w, c2_batch.shape)
        weights_batch = w

    residuals = np.asarray(
        compute_multi_angle_residuals(
            jnp.asarray(full),
            jnp.asarray(t),
            float(model.q),
            float(model.dt),
            jnp.asarray(np.asarray(phi_angles, dtype=np.float64)),
            c2_batch,
            weights_batch,
            jnp.asarray(np.asarray(contrast_per_angle, dtype=np.float64)),
            jnp.asarray(np.asarray(offset_per_angle, dtype=np.float64)),
        )
    )
    n_time = int(c2_data.shape[1])
    n_per_angle = (n_time - 1) * (n_time - 2)
    chi2_per_angle = _decompose_chi2_per_angle(
        final_residual=residuals, n_phi=n_phi, n_per_angle=n_per_angle
    )
    chi_squared = float(np.sum(residuals**2))
    return chi_squared, chi2_per_angle


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------
def fit_two_component_via_engine(
    model: HeterodyneModel,
    c2: np.ndarray,
    phi: np.ndarray,
    config: NLSQConfig,
    weights: np.ndarray | None = None,
) -> OptimizationResult:
    """Fit ``two_component`` through the shared engine, returning a contract result.

    Parameters
    ----------
    model
        :class:`HeterodyneModel` (``two_component``) at the config-default x0.
    c2
        Correlation data, shape ``(n_phi, N, N)``.
    phi
        Detector angles (degrees), shape ``(n_phi,)``.
    config
        :class:`NLSQConfig`. ``per_angle_mode`` is resolved to an engine-layout
        mode via :func:`_resolve_effective_mode` + :data:`PRODUCTION_TO_ENGINE_MODE`.
        Solver fields (``method`` / ``loss`` / tolerances / ``x_scale`` /
        ``max_nfev``) mirror the production joint fit.
    weights
        Optional weight stack matching ``c2`` shape, or ``None`` for unit weights.

    Returns
    -------
    OptimizationResult
        Physics-first ``parameters`` (``[physics | scaling_tail]``), production
        ``chi_squared`` / ``chi2_per_angle`` (SSR-conserving), covariance on the
        optimizer-DOF side, and a symmetric heterodyne ``nlsq_diagnostics`` dict.

    Raises
    ------
    NotImplementedError
        For the ``fourier`` per-angle mode (out of scope for #16a; #16b keeps it
        on the existing path).
    """
    import jax.numpy as jnp

    from xpcsjax.optimization.nlsq.heterodyne_adapter import NLSQAdapter
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig as _NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import (
        _build_heterodyne_diagnostics,
        _resolve_effective_mode,
    )
    from xpcsjax.optimization.nlsq.heterodyne_data_prep import (
        noise_normalized_reduced_chi2,
    )
    from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
        build_heterodyne_stratified_data,
    )
    from xpcsjax.optimization.nlsq.results import OptimizationResult
    from xpcsjax.optimization.nlsq.strategies.heterodyne_hybrid_streaming import (
        build_heterodyne_pointwise_model,
    )
    from xpcsjax.optimization.nlsq.strategies.stratified_ls import (
        create_stratified_chunks,
    )

    t_start = time.perf_counter()

    phi_arr = np.asarray(phi, dtype=np.float64)
    n_phi = len(phi_arr)
    phys_names = list(model.param_manager.varying_names)
    n_varying = len(phys_names)
    t = np.asarray(model.t, dtype=np.float64)
    q, dt = float(model.q), float(model.dt)

    # -- Normalize c2 / weights to the 3-D (n_phi, N, N) contract -----------
    # The public dispatcher and the old fit_nlsq_multi_phi path accept a 2-D
    # single-angle c2 matrix and add a leading axis (heterodyne_core.py:694).
    # Mirror that here so the production-support scorer
    # (compute_multi_angle_residuals, which vmaps over axis 0) never treats the
    # leading TIME dimension as the angle batch — passing a raw 2-D array would
    # otherwise raise "vmap got inconsistent sizes" against the length-1 phi /
    # contrast / offset and silently fall back on the dispatcher's best-effort.
    c2 = np.asarray(c2, dtype=np.float64)
    if c2.ndim == 2:
        c2 = c2[np.newaxis, ...]
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim == 2:
            weights = weights[np.newaxis, ...]

    # -- Resolve mode: production token -> engine-layout token --------------
    production_mode = _resolve_effective_mode(config, n_phi)
    if production_mode == "fourier":
        raise NotImplementedError(
            "fit_two_component_via_engine does not support per_angle_mode "
            "'fourier' (Task #16a scope is fixed_constant / individual / "
            "auto_averaged). Use the existing fit_nlsq_multi_phi path for fourier."
        )
    mode = PRODUCTION_TO_ENGINE_MODE[production_mode]
    assert mode in IN_SCOPE_MODES, f"resolved engine mode {mode!r} not in {IN_SCOPE_MODES}"

    # -- Build frame-0-excluded engine chunks -------------------------------
    strat_full = build_heterodyne_stratified_data(model, c2, phi_arr)
    strat = _drop_frame0_stratified_data(strat_full, t=t, n_phi=n_phi)
    chunked = create_stratified_chunks(strat, target_chunk_size=100_000)

    # -- x0 (physics-first) from the pointwise builder ----------------------
    _mf, _x, _y, p0, meta = build_heterodyne_pointwise_model(
        stratified_data=strat_full,
        model=model,
        physical_param_names=phys_names,
        per_angle_mode=mode,
    )
    p0 = np.asarray(p0, dtype=np.float64)

    # -- Frozen scaling for fixed_constant: production quantile estimator ---
    if mode == "fixed_constant":
        contrast_frozen, offset_frozen = _quantile_frozen_scaling(model, c2, n_phi)
    else:
        contrast_frozen = np.asarray(meta["contrast_arr"], dtype=np.float64)
        offset_frozen = np.asarray(meta["offset_arr"], dtype=np.float64)

    # -- Bounds (physics-first) + engine ------------------------------------
    physics_lower, physics_upper = model.param_manager.get_bounds()
    lb_pf, ub_pf = _physics_first_bounds(
        mode=mode, n_phi=n_phi, physics_lower=physics_lower, physics_upper=physics_upper
    )
    engine = _build_engine(
        mode=mode,
        chunked=chunked,
        phys_names=phys_names,
        contrast_arr=contrast_frozen,
        offset_arr=offset_frozen,
        q=q,
        dt=dt,
    )

    # -- Optimizer-vector layout (the DOF the solver actually varies) -------
    if mode == "auto_averaged":
        # COMPRESSED averaged contract: the optimizer varies EXACTLY 2 scaling
        # DOF ([physics | c_avg | o_avg]) — production's averaged DOF count
        # (_fit_joint_averaged_multi_phi). wrap_engine_averaged_residual
        # broadcasts those 2 scalars to the engine's 2*n_phi scaling-first layout
        # INSIDE the JIT residual, so the optimizer never fits independent
        # per-angle scaling. Driving the engine directly with the broadcast x0
        # (n_varying + 2*n_phi DOF) would over-parameterize the averaged fit and
        # then discard all but angle-0's fitted scalar — an inconsistent result.
        # p0 and the physics-first bounds are already the compressed form.
        x0_opt = np.asarray(p0, dtype=np.float64)
        lb_opt = np.asarray(lb_pf, dtype=np.float64)
        ub_opt = np.asarray(ub_pf, dtype=np.float64)
        wrapped = wrap_engine_averaged_residual(
            engine, n_physics=n_varying, n_phi=n_phi
        )

        def residual_fn(x: np.ndarray) -> Any:
            return wrapped(jnp.asarray(x, dtype=jnp.float64))
    else:
        # fixed_constant (identity) / individual (block permutation): pure layout
        # permutations onto the engine scaling-first vector — no DOF change.
        x0_opt = physics_first_to_scaling_first(p0, n_physics=n_varying, mode=mode, n_phi=n_phi)
        lb_opt = physics_first_to_scaling_first(lb_pf, n_physics=n_varying, mode=mode, n_phi=n_phi)
        ub_opt = physics_first_to_scaling_first(ub_pf, n_physics=n_varying, mode=mode, n_phi=n_phi)

        def residual_fn(x: np.ndarray) -> Any:
            return engine(jnp.asarray(x, dtype=jnp.float64))

    # -- NLSQ solve (SAME settings production uses) -------------------------
    joint_cfg = _NLSQConfig(
        method=config.method,
        loss=config.loss,
        ftol=config.ftol,
        xtol=config.xtol,
        gtol=config.gtol,
        x_scale=config.x_scale,
        max_nfev=config.max_nfev * n_phi,
        n_params=len(x0_opt),
    )
    adapter = NLSQAdapter(parameter_names=[f"p{i}" for i in range(len(x0_opt))])
    res = adapter.fit(
        residual_fn=residual_fn,
        initial_params=x0_opt,
        bounds=(lb_opt, ub_opt),
        config=joint_cfg,
    )
    popt_opt = np.asarray(res.parameters, dtype=np.float64)
    wall_time = time.perf_counter() - t_start

    # -- Convert popt -> physics-first; recover per-angle scaling -----------
    if mode == "auto_averaged":
        # The optimizer already varied the compressed physics-first vector;
        # identity passthrough (validates length) — no un-permutation needed.
        popt_pf = engine_popt_to_compressed_averaged(popt_opt, n_physics=n_varying)
    else:
        popt_pf = scaling_first_to_physics_first(
            popt_opt, n_physics=n_varying, mode=mode, n_phi=n_phi
        )
    physics_fitted = popt_pf[:n_varying]

    if mode == "fixed_constant":
        contrast_used = contrast_frozen
        offset_used = offset_frozen
    elif mode == "auto_averaged":
        c_avg = float(popt_pf[n_varying])
        o_avg = float(popt_pf[n_varying + 1])
        contrast_used = np.full(n_phi, c_avg, dtype=np.float64)
        offset_used = np.full(n_phi, o_avg, dtype=np.float64)
    else:  # individual
        contrast_used = np.asarray(popt_pf[n_varying : n_varying + n_phi], dtype=np.float64)
        offset_used = np.asarray(popt_pf[n_varying + n_phi :], dtype=np.float64)

    # -- Production-support objective (REUSE production residual + helpers) --
    chi_squared, chi2_per_angle = _production_support_chi2(
        model=model,
        c2_data=c2,
        phi_angles=phi_arr,
        physics_varying=physics_fitted,
        contrast_per_angle=contrast_used,
        offset_per_angle=offset_used,
        weights=weights,
    )

    n_total_params = int(popt_pf.size)
    data_valid = n_phi * (int(c2.shape[1]) - 1) * (int(c2.shape[1]) - 2)
    reduced_chi2 = noise_normalized_reduced_chi2(
        ssr=chi_squared,
        c2_data=np.asarray(c2, dtype=np.float64),
        n_data_valid=int(data_valid),
        n_params=n_total_params,
    )

    # -- Covariance / uncertainties (optimizer-DOF side) --------------------
    # For fixed_constant / individual the optimizer DOF == popt_pf length (the
    # scaling-first <-> physics-first map is a permutation). For auto_averaged
    # the optimizer DOF == n_varying + 2 (the 2 compressed averaged scalars),
    # which also equals popt_pf length — so res.covariance is dimensionally
    # the physics-first compressed covariance and passes through unchanged
    # (Task-2.2: a 2->2*n_phi averaged covariance permutation is undefined).
    if res.covariance is not None and np.asarray(res.covariance).shape == (
        n_total_params,
        n_total_params,
    ):
        covariance = np.asarray(res.covariance, dtype=np.float64)
    else:
        covariance = np.full((n_total_params, n_total_params), np.nan, dtype=np.float64)
    if res.uncertainties is not None and np.asarray(res.uncertainties).shape == (
        n_total_params,
    ):
        uncertainties = np.asarray(res.uncertainties, dtype=np.float64)
    else:
        uncertainties = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    convergence_status = "converged" if res.success else "failed"
    quality_flag = "good" if res.success else "marginal"

    # ``n_physics``: mirror production EXACTLY. The constant path
    # (``_fit_joint_constant_multi_phi``) does NOT pass ``n_physics`` to
    # ``OptimizationResult`` (defaults to ``None``); the averaged / individual
    # paths set it to the physics-varying count. Matching this keeps the engine
    # result byte-for-byte contract-equal to the production path it mirrors.
    n_physics_field = None if mode == "fixed_constant" else int(n_varying)

    # -- Per-mode nlsq_diagnostics (mirror the production path's extras) ----
    joint_param_names = _engine_joint_param_names(mode, phys_names, n_phi)
    diagnostics = _assemble_diagnostics(
        mode=mode,
        chi2_per_angle=chi2_per_angle,
        joint_param_names=joint_param_names,
        phi_angles=phi_arr,
        n_phi=n_phi,
        contrast_used=contrast_used,
        offset_used=offset_used,
        contrast_quantile=np.asarray(meta["contrast_arr"], dtype=np.float64),
        offset_quantile=np.asarray(meta["offset_arr"], dtype=np.float64),
        res=res,
        wall_time=wall_time,
        build_diag=_build_heterodyne_diagnostics,
    )

    return OptimizationResult(
        parameters=np.asarray(popt_pf, dtype=np.float64),
        uncertainties=uncertainties,
        covariance=covariance,
        chi_squared=chi_squared,
        reduced_chi_squared=reduced_chi2,
        convergence_status=convergence_status,
        iterations=int(res.n_iterations or 0),
        execution_time=wall_time,
        device_info={"backend": "cpu", "adapter": "engine_route.StratifiedResidualFunctionJIT"},
        recovery_actions=[],
        quality_flag=quality_flag,
        streaming_diagnostics=None,
        stratification_diagnostics=None,
        nlsq_diagnostics=diagnostics,
        n_physics=n_physics_field,
    )


def _engine_joint_param_names(
    mode: str, phys_names: list[str], n_phi: int
) -> list[str]:
    """Physics-first joint parameter-name list matching ``parameters`` length."""
    if mode == "fixed_constant":
        return list(phys_names)
    if mode == "auto_averaged":
        return [*phys_names, "contrast_avg", "offset_avg"]
    # individual
    contrast_names = [f"contrast_{i}" for i in range(n_phi)]
    offset_names = [f"offset_{i}" for i in range(n_phi)]
    return [*phys_names, *contrast_names, *offset_names]


def _assemble_diagnostics(
    *,
    mode: str,
    chi2_per_angle: np.ndarray,
    joint_param_names: list[str],
    phi_angles: np.ndarray,
    n_phi: int,
    contrast_used: np.ndarray,
    offset_used: np.ndarray,
    contrast_quantile: np.ndarray,
    offset_quantile: np.ndarray,
    res: Any,
    wall_time: float,
    build_diag: Any,
) -> dict[str, Any]:
    """Build the per-mode ``nlsq_diagnostics`` dict via the production helper.

    Mirrors the ``_build_heterodyne_diagnostics`` call each production path makes:

    * ``fixed_constant`` ↔ ``_fit_joint_constant_multi_phi``
      (``scaling_source="quantile_fixed"``, ``*_per_angle_fixed``).
    * ``auto_averaged`` ↔ ``_fit_joint_averaged_multi_phi``
      (``scaling_source="averaged_then_fitted"``, ``averaged_*``).
    * ``individual`` ↔ ``_fit_joint_multi_phi``
      (``scaling_source="fitted"``, ``*_per_angle_fitted``).

    Anti-degeneracy layers (L2/L3/L4) are not run on this build-alongside path
    (the standard production heterodyne path runs no in-memory L2/L3 either), so
    the activation flags default to ``False`` — the symmetric activation keys are
    still emitted by the shared assembler.
    """
    # ``common`` carries the keys EVERY mode emits. ``phi_angles`` /
    # ``n_angles_joint`` are emitted by the averaged / individual joint paths
    # but NOT by the constant path (its hand-rolled dict omits them), so they
    # are added per-branch below rather than in ``common``.
    common = {
        "chi2_per_angle": chi2_per_angle,
        "fourier_basis_dim": None,
        "parameter_names": list(joint_param_names),
        "convergence_reason": str(res.convergence_reason),
        "n_function_evals": int(res.n_function_evals or 0),
        "n_iterations": int(res.n_iterations or 0),
        "wall_time_seconds": float(wall_time),
        "message": str(res.message),
    }
    joint_extras = {
        "phi_angles": np.asarray(phi_angles, dtype=np.float64),
        "n_angles_joint": int(n_phi),
    }

    if mode == "fixed_constant":
        return build_diag(
            per_angle_mode="constant",
            scaling_source="quantile_fixed",
            contrast_per_angle_fixed=np.asarray(contrast_used, dtype=np.float64),
            offset_per_angle_fixed=np.asarray(offset_used, dtype=np.float64),
            **common,
        )
    if mode == "auto_averaged":
        return build_diag(
            per_angle_mode="averaged",
            scaling_source="averaged_then_fitted",
            averaged_contrast=float(contrast_used[0]),
            averaged_offset=float(offset_used[0]),
            contrast_per_angle_quantile=np.asarray(contrast_quantile, dtype=np.float64),
            offset_per_angle_quantile=np.asarray(offset_quantile, dtype=np.float64),
            contrast_initial_average=float(np.mean(contrast_quantile)),
            offset_initial_average=float(np.mean(offset_quantile)),
            **common,
            **joint_extras,
        )
    # individual
    return build_diag(
        per_angle_mode="individual",
        scaling_source="fitted",
        contrast_per_angle_fitted=np.asarray(contrast_used, dtype=np.float64),
        offset_per_angle_fitted=np.asarray(offset_used, dtype=np.float64),
        **common,
        **joint_extras,
    )
