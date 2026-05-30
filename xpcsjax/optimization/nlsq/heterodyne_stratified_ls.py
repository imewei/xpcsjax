"""Heterodyne stratified least-squares solver.

Mirrors the homodyne stratified-LS path (strategies/stratified_ls.py) for the
heterodyne two_component model. Reuses the model-agnostic chunking helpers and
adds a heterodyne-specific joint pointwise residual whose per-angle scaling is
expanded from the varying parameter vector each iteration.

Parameter packing is physics-first ([physics | scaling]) to match the rest of
the heterodyne result handling. The objective equals the in-memory joint fit's
objective; the only intended behavioral change is the seed-42 pre-shuffle.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from xpcsjax.optimization.nlsq.strategies.chunking import (
    compute_stratification_diagnostics,
    create_angle_stratified_indices,
    estimate_stratification_memory,
)

_SHUFFLE_SEED = 42


def reorder_for_stratification(
    phi_flat: np.ndarray,
    target_chunk_size: int = 100_000,
    *,
    shuffle: bool = True,
    seed: int = _SHUFFLE_SEED,
) -> tuple[np.ndarray, list[int]]:
    """Return a permutation that angle-stratifies (and optionally shuffles) points.

    Parameters
    ----------
    phi_flat : np.ndarray
        Per-point angle labels, shape ``(N,)``.
    target_chunk_size : int
        Interleaved-stratification chunk target (model-agnostic, from chunking.py).
    shuffle : bool
        If True, apply a fixed-seed PRE-shuffle to the flat point order BEFORE
        stratification, then compose back. Stratification is re-derived from the
        relabeled angles, so each chunk keeps its balanced angle multiset; only
        WHICH concrete points fill each angle's slots changes (homodyne
        local-minimum-avoidance parity — alters trajectory, not objective). With
        ``shuffle=False`` the behavior is identical to no shuffle (seed-independent).
    seed : int
        Pre-shuffle seed (fixed at 42 for reproducibility; matches homodyne).

    Returns
    -------
    (perm, chunk_sizes) : tuple[np.ndarray, list[int]]
        ``perm`` reorders any per-point array; ``chunk_sizes`` are the
        interleaved chunk sizes from stratification.
    """
    phi_flat = np.asarray(phi_flat)
    n = len(phi_flat)
    if shuffle:
        rng = np.random.RandomState(seed)
        pre = rng.permutation(n)  # pre-shuffle the flat point order
    else:
        pre = np.arange(n)
    # Stratify the (pre-shuffled) labels, then compose back so chunk balance is
    # preserved. ``strat_perm`` indexes ``phi_flat[pre]``, so ``pre[strat_perm]``
    # maps back to the original point indices.
    strat_perm, chunk_sizes = create_angle_stratified_indices(phi_flat[pre], target_chunk_size)
    perm = pre[np.asarray(strat_perm, dtype=np.int64)]
    return perm, list(chunk_sizes)


def make_scaling_expander(
    per_angle_mode: str,
    n_phi: int,
    *,
    fourier: Any | None = None,
) -> tuple[Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]], int]:
    """Return ``(expander, n_scaling_params)`` for the active per-angle mode.

    ``expander(scaling_params) -> (contrast[n_phi], offset[n_phi])`` maps the
    varying scaling parameters to per-angle contrast/offset arrays. Physics-first
    packing means these scaling params are the TAIL of the joint vector.

    - averaged: 2 params (one contrast, one offset) broadcast to all angles.
    - individual: 2*n_phi params (contrast block then offset block).
    - fourier: 2*(2K+1) Fourier coefficients via ``fourier``.

    ``constant`` and any unrecognized mode are unsupported by stratified-LS and
    raise ``NotImplementedError`` (the dispatch gate falls back to the in-memory
    joint fit).
    """
    if per_angle_mode == "averaged":

        def expand(s: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            return jnp.full((n_phi,), s[0]), jnp.full((n_phi,), s[1])

        return expand, 2

    if per_angle_mode == "individual":

        def expand(s: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            return s[:n_phi], s[n_phi:2 * n_phi]

        return expand, 2 * n_phi

    if per_angle_mode == "fourier":
        if fourier is None:
            raise ValueError(
                "fourier mode requires a FourierReparameterizer (fourier=...)"
            )
        # The scaling vector IS the full Fourier coefficient vector
        # [contrast_coeffs (n_coeffs_per_param) | offset_coeffs (n_coeffs_per_param)].
        # fourier_to_per_angle_jax splits and maps both halves to per-angle
        # arrays in one JIT-safe call — identical to the conversion done every
        # iteration by ``_fit_joint_multi_phi`` in heterodyne_core.py.
        n_scaling = int(fourier.n_coeffs)

        def expand(s: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            return fourier.fourier_to_per_angle_jax(s)

        return expand, n_scaling

    raise NotImplementedError(
        f"stratified-LS does not support per_angle_mode={per_angle_mode!r} "
        "(supported: averaged, individual, fourier)"
    )


def build_joint_pointwise_residual(
    *,
    model: Any,
    stratified_data: Any,
    per_angle_mode: str,
    init_scaling: np.ndarray,
    fourier: Any | None = None,
    perm: np.ndarray | None = None,
) -> tuple[Callable[[np.ndarray], jnp.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a flat pointwise residual with VARYING per-angle scaling.

    Unlike :func:`build_heterodyne_pointwise_model` (which bakes the per-angle
    contrast/offset in as FIXED quantile arrays), this residual expands the
    per-angle scaling from the TAIL of the joint parameter vector each
    iteration via ``make_scaling_expander``. Parameter packing is physics-first:
    ``params = [physics (n_physics) | scaling (n_scaling)]``.

    Parameters
    ----------
    model :
        Configured ``HeterodyneModel`` (provides ``param_manager``, ``q``, ``dt``).
    stratified_data :
        Flat heterodyne data from ``build_heterodyne_stratified_data``.
    per_angle_mode :
        One of the modes accepted by ``make_scaling_expander``
        (``"averaged"`` / ``"individual"`` / ``"fourier"``).
    init_scaling :
        Mode-appropriate initial scaling tail seed (the driver computes this).
        Length must equal ``n_scaling`` for the active mode: ``2`` for
        averaged, ``2*n_phi`` for individual, ``fourier.n_coeffs`` for fourier.
    fourier :
        Optional Fourier descriptor passed through to ``make_scaling_expander``.
    perm :
        Optional permutation of the flat support (objective-invariant reorder /
        shuffle used by the stratified-LS path). ``None`` keeps native order.

    Returns
    -------
    residual_fn : callable
        ``residual_fn(params) -> jnp.ndarray`` of length ``meta["n_data_points"]``.
    x_data : np.ndarray, (N, 3) int32
        ``[phi_idx, t1_idx, t2_idx]`` per point (post-``perm`` if given).
    y_data : np.ndarray, (N,) float64
        Observed C2 values (post-``perm`` if given).
    p0_full : np.ndarray, (n_physics + n_scaling,) float64
        Physics-first initial joint vector.
    meta : dict
        ``build_heterodyne_pointwise_model``'s meta plus
        ``{"n_physics", "n_phi", "n_scaling"}``.
    """
    from xpcsjax.core.heterodyne_jax_backend import compute_c2_heterodyne_pointwise
    from xpcsjax.optimization.nlsq.strategies.heterodyne_hybrid_streaming import (
        build_heterodyne_pointwise_model,
    )

    physical_param_names = list(model.param_manager.varying_names)
    # discard the fixed-scaling model_fn; we re-derive the residual below with
    # scaling lifted into the varying params.
    _fixed_fn, x_data, y_data, _p0_physics, meta = build_heterodyne_pointwise_model(
        stratified_data=stratified_data,
        model=model,
        physical_param_names=physical_param_names,
    )
    sigma = meta.get("sigma")

    # Optional reorder/shuffle of the flat support (objective-invariant).
    if perm is not None:
        perm = np.asarray(perm, dtype=np.int64)
        x_data = x_data[perm]
        y_data = y_data[perm]
        if sigma is not None:
            sigma = np.asarray(sigma)[perm]

    n_physics = int(model.param_manager.n_varying)
    n_phi = int(np.asarray(meta["phi_unique"]).shape[0])
    expander, n_scaling = make_scaling_expander(per_angle_mode, n_phi, fourier=fourier)

    fixed_full_jax = jnp.asarray(model.param_manager.get_full_values(), dtype=jnp.float64)
    varying_indices_jax = jnp.array(
        list(model.param_manager.varying_indices), dtype=jnp.int32
    )
    # Use the SAME time grid the pointwise kernel was indexed against (the
    # t1_idx/t2_idx in x_data address THIS array, not necessarily model.t).
    t_jax = jnp.asarray(meta["t_unique"], dtype=jnp.float64)
    q_val = float(model.q)
    dt_val = float(model.dt)
    phi_unique_jax = jnp.asarray(meta["phi_unique"], dtype=jnp.float64)
    x_jax = jnp.asarray(x_data, dtype=jnp.int32)
    y_jax = jnp.asarray(y_data, dtype=jnp.float64)
    inv_sigma_jax = (
        jnp.asarray(1.0 / np.asarray(sigma, dtype=np.float64), dtype=jnp.float64)
        if sigma is not None
        else None
    )

    @jax.jit
    def residual_fn(params: np.ndarray) -> jnp.ndarray:
        params = jnp.asarray(params, dtype=jnp.float64)
        physics = params[:n_physics]
        scaling = params[n_physics:]
        contrast, offset = expander(scaling)
        full = fixed_full_jax.at[varying_indices_jax].set(physics)
        phi_idx = x_jax[:, 0]
        t1_idx = x_jax[:, 1]
        t2_idx = x_jax[:, 2]
        model_vals = compute_c2_heterodyne_pointwise(
            full,
            t_jax,
            q_val,
            dt_val,
            phi_unique=phi_unique_jax,
            phi_idx=phi_idx,
            t1_idx=t1_idx,
            t2_idx=t2_idx,
            contrast=contrast,
            offset=offset,
        )
        resid = jnp.squeeze(model_vals) - y_jax
        if inv_sigma_jax is not None:
            resid = resid * inv_sigma_jax
        return resid

    init_scaling = np.asarray(init_scaling, dtype=np.float64)
    if init_scaling.shape[0] != n_scaling:
        raise ValueError(
            f"init_scaling has length {init_scaling.shape[0]} but per_angle_mode="
            f"{per_angle_mode!r} (n_phi={n_phi}) requires n_scaling={n_scaling}"
        )
    p0_full = np.concatenate(
        [
            np.asarray(model.param_manager.get_initial_values(), dtype=np.float64),
            init_scaling,
        ]
    )
    out_meta = {**meta, "n_physics": n_physics, "n_phi": n_phi, "n_scaling": n_scaling}
    return residual_fn, x_data, y_data, p0_full, out_meta


def fit_heterodyne_stratified_least_squares(
    *,
    model: Any,
    c2: np.ndarray,
    phi: np.ndarray,
    config: Any,
    weights: np.ndarray | None,
    target_chunk_size: int = 100_000,
    shuffle: bool = True,
    use_index_based: bool = True,
    check_memory_safety: bool = True,
) -> Any:
    """Mode-aware heterodyne stratified-LS solve. Returns OptimizationResult.

    Resolves the effective per-angle mode (``averaged`` / ``fourier``) via
    :func:`_resolve_effective_mode`, computes the mode-appropriate scaling-tail
    seed from per-angle quantiles, and runs a single joint pointwise
    least-squares solve. The objective equals the in-memory joint fit for the
    same mode; the only behavioral change is the optional seed-42 reorder/shuffle
    of the flat point support (objective-invariant — reordering residual elements
    does not change the sum of squares).

    Only the JOINT modes ``averaged`` and ``fourier`` are supported. ``individual``
    (sequential per-angle) and ``constant`` (frozen scaling) raise
    ``NotImplementedError``; the dispatch gate in ``__init__.py`` only routes
    averaged/fourier here and additionally wraps this driver in a best-effort
    try/except that falls through to the in-memory joint fit.

    Parameters
    ----------
    use_index_based :
        Threaded into ``compute_stratification_diagnostics`` and
        ``estimate_stratification_memory``. Heterodyne is structurally
        index-based (the pointwise kernel addresses a flat support by integer
        index), so the value is informational — but it is sourced from config,
        not a literal, so the recorded diagnostic reflects the user's setting.
    check_memory_safety :
        When True, the memory estimate's ``is_safe`` flag is consulted and a
        warning is logged if the projected peak exceeds the safe fraction of
        RAM. Best-effort and non-fatal. When False, the estimate is still
        computed for diagnostics but the safety warning is suppressed.
    """
    from xpcsjax.optimization.nlsq.heterodyne_adapter import NLSQAdapter
    from xpcsjax.optimization.nlsq.heterodyne_core import _resolve_effective_mode
    from xpcsjax.optimization.nlsq.heterodyne_result_builder import (
        build_hybrid_streaming_result,
    )
    from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
        build_heterodyne_stratified_data,
    )
    from xpcsjax.optimization.nlsq.parameter_utils import (
        compute_quantile_per_angle_scaling,
    )

    strat = build_heterodyne_stratified_data(model, c2, phi, weights)
    n_phi = len(phi)
    mode = _resolve_effective_mode(config, n_phi)

    # Defensive scope gate (belt-and-suspenders for the dispatch gate in
    # __init__.py): only the JOINT modes ``averaged`` and ``fourier`` use
    # stratified-LS. ``individual`` is sequential per-angle (a different
    # objective) and ``constant`` freezes scaling — both must use the in-memory
    # path, so the driver refuses to run them even if called directly.
    if mode not in ("averaged", "fourier"):
        raise NotImplementedError(
            f"stratified-LS supports per_angle_mode in ('averaged', 'fourier'); "
            f"got resolved mode={mode!r} (individual is sequential per-angle; "
            "constant freezes scaling — both use the in-memory joint path)"
        )

    contrast_pa, offset_pa = compute_quantile_per_angle_scaling(strat)
    contrast_pa = np.asarray(contrast_pa, dtype=np.float64)
    offset_pa = np.asarray(offset_pa, dtype=np.float64)

    fourier: Any | None = None
    if mode == "averaged":
        init_scaling = np.array(
            [float(np.nanmean(contrast_pa)), float(np.nanmean(offset_pa))],
            dtype=np.float64,
        )
        scaling_names = ["contrast", "offset"]
    else:  # mode == "fourier" — guaranteed by the scope gate above
        from xpcsjax.optimization.nlsq.fourier_reparam import (
            FourierReparamConfig,
            FourierReparameterizer,
        )

        fourier_config = FourierReparamConfig(
            mode="fourier",
            fourier_order=config.fourier_order,
            auto_threshold=config.fourier_auto_threshold,
        )
        phi_rad = np.deg2rad(np.asarray(phi).astype(np.float64))
        fourier = FourierReparameterizer(phi_rad, fourier_config)
        # Seed coeffs from the per-angle quantiles via the least-squares inverse.
        # per_angle_to_fourier returns the full n_coeffs vector
        # [contrast_coeffs | offset_coeffs] in one call.
        init_scaling = np.asarray(
            fourier.per_angle_to_fourier(contrast_pa, offset_pa), dtype=np.float64
        )
        scaling_names = [f"fourier_{i}" for i in range(int(fourier.n_coeffs))]

    # Build the residual once (native order) to obtain the FILTERED flat support
    # (off-diagonal AND t>0 — strat.phi_flat is the full N_total grid including
    # the diagonal/t=0 boundary, so a perm over it would not index the residual
    # support). The stratification/shuffle perm is then built over that filtered
    # support and re-applied by a second build. The reorder is objective-
    # invariant (it only permutes residual elements).
    # only x_data0 is used (to derive the filtered-support perm)
    _rfn0, x_data0, _y0, _p00, _meta0 = build_joint_pointwise_residual(
        model=model,
        stratified_data=strat,
        per_angle_mode=mode,
        init_scaling=init_scaling,
        fourier=fourier,
    )
    # Stratify on the integer phi-index column directly (identity, not float
    # value) — robust regardless of how create_angle_stratified_indices bins.
    phi_idx_filtered = np.asarray(x_data0[:, 0], dtype=np.int64).astype(np.float64)
    _t0_strat = time.perf_counter()
    perm, chunk_sizes = reorder_for_stratification(
        phi_idx_filtered,
        target_chunk_size,
        shuffle=shuffle,
    )
    _execution_time_ms = (time.perf_counter() - _t0_strat) * 1000.0
    # Support-ordering contract: ``perm`` is a permutation of the FILTERED captured
    # support (``x_data0[:, 0]``). If a future builder change reorders or resizes
    # that support, this guard fails loudly instead of silently mis-indexing the
    # residual via a length-mismatched permutation.
    assert len(perm) == x_data0.shape[0], (
        f"stratification perm length ({len(perm)}) != filtered support length "
        f"({x_data0.shape[0]}); the builder's flat-support ordering changed"
    )
    residual_fn, x_data, y_data, p0_full, meta = build_joint_pointwise_residual(
        model=model,
        stratified_data=strat,
        per_angle_mode=mode,
        init_scaling=init_scaling,
        fourier=fourier,
        perm=perm,
    )
    # The reordered build must produce the SAME support length perm was derived
    # against — otherwise ``x_data = x_data0[perm]`` (inside the builder) would have
    # indexed a differently-sized array.
    assert len(perm) == x_data.shape[0], (
        f"stratification perm length ({len(perm)}) != rebuilt support length "
        f"({x_data.shape[0]}); native and permuted builds disagree on support size"
    )

    n_scaling = int(meta["n_scaling"])
    lower_phys, upper_phys = model.param_manager.get_bounds()
    if mode == "fourier":
        # Fourier coefficients are bounded per the reparameterizer (matches the
        # in-memory _fit_joint_multi_phi path, which uses fourier.get_bounds()).
        scaling_lower, scaling_upper = fourier.get_bounds()
        scaling_lower = np.asarray(scaling_lower, np.float64)
        scaling_upper = np.asarray(scaling_upper, np.float64)
    else:
        # averaged / individual: contrast and offset are non-negative.
        scaling_lower = np.zeros(n_scaling, dtype=np.float64)
        scaling_upper = np.full(n_scaling, np.inf, dtype=np.float64)
    lower = np.concatenate([np.asarray(lower_phys, np.float64), scaling_lower])
    upper = np.concatenate([np.asarray(upper_phys, np.float64), scaling_upper])

    # Full joint parameter-name list ([physics | scaling]) — used both for the
    # adapter and (Fix 4) threaded to the result builder so the diagnostics
    # ``parameter_names`` align 1:1 with the full popt length.
    joint_param_names = [*model.param_manager.varying_names, *scaling_names]
    adapter = NLSQAdapter(parameter_names=joint_param_names)
    fit = adapter.fit(
        residual_fn=residual_fn,
        initial_params=p0_full,
        bounds=(lower, upper),
        config=config,
    )

    popt = np.asarray(fit.parameters, dtype=np.float64)
    pcov = (
        np.asarray(fit.covariance, dtype=np.float64)
        if fit.covariance is not None
        else np.full((popt.size, popt.size), np.nan)
    )

    # SSR conservation: recompute the data-only residual at the solution and
    # decompose chi^2 by phi index. Mirrors the joint averaged path, which
    # reports ``chi_squared = sum(data_only_residual**2)`` rather than the
    # optimizer's robust-loss cost. We pass ``info["cost"] = 0.5 * SSR`` so the
    # builder's ``chi_squared = info["cost"] * 2`` recovers the exact SSR.
    final_residual = np.asarray(residual_fn(popt), dtype=np.float64)
    ssr = float(np.sum(final_residual**2))
    phi_idx_flat = np.asarray(x_data[:, 0], dtype=np.int64)
    n_phi_meta = int(meta["n_phi"])
    chi2_per_angle = np.zeros(n_phi_meta, dtype=np.float64)
    np.add.at(chi2_per_angle, phi_idx_flat, final_residual**2)

    # Compute stratification diagnostics and memory estimate.
    # phi_original and phi_stratified are the same length (phi_idx_filtered[perm]
    # is a permutation of phi_idx_filtered), so compute_stratification_diagnostics
    # sees matching arrays.
    phi_stratified = phi_idx_filtered[perm]
    strat_diag = compute_stratification_diagnostics(
        phi_original=phi_idx_filtered,
        phi_stratified=phi_stratified,
        execution_time_ms=_execution_time_ms,
        use_index_based=use_index_based,
        target_chunk_size=target_chunk_size,
        chunk_sizes=chunk_sizes,
    )
    mem_estimate = estimate_stratification_memory(
        n_points=int(phi_idx_filtered.shape[0]),
        use_index_based=use_index_based,
    )
    # check_memory_safety: best-effort, non-fatal warning when the projected
    # peak exceeds the safe RAM fraction (estimate_stratification_memory sets
    # ``is_safe`` from psutil). When disabled, the estimate is still recorded in
    # diagnostics but the warning is suppressed.
    if check_memory_safety and not mem_estimate.get("is_safe", True):
        from xpcsjax.utils.logging import get_logger

        get_logger(__name__).warning(
            "Heterodyne stratification memory estimate is unsafe: peak %.1f MB "
            "exceeds the safe fraction of available RAM (n_points=%d). "
            "Proceeding (non-fatal).",
            float(mem_estimate.get("peak_memory_mb", 0.0)),
            int(phi_idx_filtered.shape[0]),
        )

    info = {
        "success": bool(fit.success),
        "cost": 0.5 * ssr,
        "nit": int(fit.n_iterations or 0),
        "wall_time": float(fit.wall_time_seconds or 0.0),
        "n_data_points": int(meta["n_data_points"]),
        "stratification_memory": mem_estimate,
    }
    return build_hybrid_streaming_result(
        model=model,
        popt=popt,
        pcov=pcov,
        info=info,
        phi_angles=np.asarray(phi),
        per_angle_mode=mode,
        scaling_source="stratified_ls",
        chi2_per_angle=chi2_per_angle,
        stratification_diagnostics=strat_diag,
        parameter_names=joint_param_names,
    )
