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
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from xpcsjax.optimization.nlsq.strategies.chunking import (
    compute_stratification_diagnostics,
    create_angle_stratified_indices,
    estimate_stratification_memory,
)

if TYPE_CHECKING:
    from xpcsjax.optimization.nlsq.anti_degeneracy_controller import (
        AntiDegeneracyController,
    )

_SHUFFLE_SEED = 42

# Column-block width for the host covariance Jacobian (see _chunked_jacfwd_dense).
# n_params is small (16 for two_component); 4 blocks of 4 tangents cap the
# forward-AD tangent width at 4 instead of n_params while staying byte-identical.
_COV_JACFWD_COL_BLOCK = 4


def _chunked_jacfwd_dense(
    fn: Callable[[np.ndarray], jnp.ndarray],
    x: np.ndarray,
    *,
    col_block: int = _COV_JACFWD_COL_BLOCK,
) -> np.ndarray:
    """Column-blocked forward-mode Jacobian, numerically identical to ``jax.jacfwd``.

    ``jax.jacfwd(fn)(x)`` builds the full ``(n_out, n_in)`` Jacobian by pushing
    all ``n_in`` basis tangents through ``fn`` at once, so every intermediate of
    ``fn`` is materialised at width ``n_in``. For the heterodyne stratified-LS
    covariance ``n_out`` is the full support (~23M points) and ``n_in`` is the
    16-ish joint parameters, so that ``n_in``-wide tangent is the dominant
    transient — the ~3 GB+ spike that drives the post-solve memory peak.

    Computing the columns in small blocks (each a ``vmap``'d JVP over ``col_block``
    basis vectors, moved to host and released before the next block) yields the
    SAME columns — ``jvp`` is exact and column order is preserved — while capping
    the live tangent width at ``col_block``. The assembled ``J`` (and therefore
    ``JᵀJ`` and the covariance) is byte-identical to ``jax.jacfwd`` up to XLA
    fusion noise (≤ ULP); this only affects the post-solve covariance, never the
    fit trajectory.

    Parameters
    ----------
    fn : callable
        ``params (n_in,) -> residuals (n_out,)`` (the joint residual). May be
        ``jax.jit``-wrapped.
    x : np.ndarray
        Point at which to evaluate the Jacobian (the converged ``popt``).
    col_block : int, optional
        Number of parameter columns evaluated per block. Defaults to
        :data:`_COV_JACFWD_COL_BLOCK`.

    Returns
    -------
    np.ndarray, (n_out, n_in) float64
        The dense Jacobian, matching ``np.asarray(jax.jacfwd(fn)(x))``.
    """
    x_jax = jnp.asarray(x, dtype=jnp.float64)
    n_in = int(x_jax.shape[0])
    eye = jnp.eye(n_in, dtype=x_jax.dtype)

    def _jvp_col(tangent: jnp.ndarray) -> jnp.ndarray:
        # jvp(fn, primal, e_j)[1] == d fn / d x_j == column j of the Jacobian.
        return jax.jvp(fn, (x_jax,), (tangent,))[1]

    blocks: list[np.ndarray] = []
    for c0 in range(0, n_in, max(1, col_block)):
        tangents = eye[c0 : c0 + max(1, col_block)]  # (b, n_in)
        # (b, n_out): row r is column (c0 + r) of J. Pull to host and let the
        # device buffer for this block free before the next block allocates.
        block_cols = np.asarray(jax.vmap(_jvp_col)(tangents), dtype=np.float64)
        blocks.append(block_cols)

    # Stack column-blocks -> (n_in, n_out), transpose -> (n_out, n_in).
    return np.concatenate(blocks, axis=0).T


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


def _emit_anti_degeneracy_parity_banners(
    *,
    anti_degeneracy_dict: dict | None,
    phi_deg: np.ndarray,
    n_physical: int,
) -> AntiDegeneracyController | None:
    """Instantiate the shared AntiDegeneracyController for laminar-parity banners.

    Mirrors laminar's ``fit_with_stratified_least_squares``: instantiating the
    controller emits the ``ANTI-DEGENERACY: Layer 2/3/4`` + mode setup banners
    and builds the L2/L3/L4 components per the config enable flags. L5 is gated
    off for ``two_component`` by ``_LAYER_GATES``. This is purely for log/
    diagnostic-surface parity — the numeric single-solve path is unchanged, and
    the flat ``hierarchical_active`` / ``regularization_active`` markers stay
    ``False`` exactly as laminar reports them on its stratified path.

    Best-effort: returns the controller, or ``None`` if no config is provided or
    construction fails (a banner failure must never break the fit).
    """
    if not isinstance(anti_degeneracy_dict, dict) or not anti_degeneracy_dict:
        return None
    try:
        from xpcsjax.config.parameter_registry import AnalysisMode
        from xpcsjax.optimization.nlsq import heterodyne_logging as _hlog
        from xpcsjax.optimization.nlsq.anti_degeneracy_controller import (
            AntiDegeneracyController,
        )

        # Frame the controller's "Enabled: True" Layer 2/3/4 banners (logged on
        # construction below) as CONFIGURATION, so they are not misread as
        # contradicting the honest [AS EXECUTED] summary at fit end.
        _hlog.log_configured_layers_preamble()

        phi_rad = np.deg2rad(np.asarray(phi_deg, dtype=np.float64))
        return AntiDegeneracyController.from_config(
            config_dict=anti_degeneracy_dict,
            n_phi=int(phi_rad.shape[0]),
            phi_angles=phi_rad,
            n_physical=int(n_physical),
            per_angle_scaling=True,
            is_laminar_flow=False,
            analysis_mode=AnalysisMode.TWO_COMPONENT,
        )
    except Exception as exc:  # best-effort: banners must never break a fit
        from xpcsjax.utils.logging import get_logger as _get_logger

        _get_logger(__name__).warning(
            "Anti-degeneracy parity banners skipped (controller init failed: %s)", exc
        )
        return None


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
            return s[:n_phi], s[n_phi : 2 * n_phi]

        return expand, 2 * n_phi

    if per_angle_mode == "fourier":
        if fourier is None:
            raise ValueError("fourier mode requires a FourierReparameterizer (fourier=...)")
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
    varying_indices_jax = jnp.array(list(model.param_manager.varying_indices), dtype=jnp.int32)
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
        # Use a distinct local for the JAX-converted vector so the numpy-typed
        # ``params`` argument is not reassigned to a jnp Array (keeps mypy happy
        # at the numpy/JAX boundary without changing behavior).
        p = jnp.asarray(params, dtype=jnp.float64)
        physics = p[:n_physics]
        scaling = p[n_physics:]
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


def _reconstruct_per_angle_scaling(
    params_native: Any,
    *,
    mode: str,
    n_physics: int,
    n_phi: int,
    fourier: Any | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(contrast[n_phi], offset[n_phi])`` from a native ``[physics | scaling]`` vector.

    L3 must constrain the *per-angle* scaling CV — the physically meaningful
    quantity — for BOTH ``individual`` and ``fourier`` modes. For ``individual``
    the scaling tail IS the per-angle blocks; for ``fourier`` the per-angle arrays
    are reconstructed from the coefficients via ``fourier_to_per_angle_jax``
    (mirroring ``heterodyne_core``'s row-append Fourier path — regularizing the raw
    coefficient blocks is wrong because coefficient variance can be smooth while
    reconstructed per-angle contrast/offset still vary).
    """
    tail = params_native[n_physics:]
    if mode == "individual":
        return tail[:n_phi], tail[n_phi : 2 * n_phi]
    if mode == "fourier":
        if fourier is None:
            raise ValueError("fourier mode L3 requires a FourierReparameterizer")
        return fourier.fourier_to_per_angle_jax(tail)
    # averaged: a single scalar per group broadcast to all angles
    return jnp.full((n_phi,), tail[0]), jnp.full((n_phi,), tail[1])


def _per_angle_cv(contrasts: jnp.ndarray, offsets: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(contrast_CV, offset_CV)`` with a safe-divide on near-zero means."""
    c_mean = jnp.mean(contrasts)
    c_cv = jnp.where(
        jnp.abs(c_mean) > 1e-10, jnp.std(contrasts) / jnp.abs(c_mean), jnp.std(contrasts)
    )
    o_mean = jnp.mean(offsets)
    o_cv = jnp.where(
        jnp.abs(o_mean) > 1e-10, jnp.std(offsets) / jnp.abs(o_mean), jnp.std(offsets)
    )
    return c_cv, o_cv


def _run_hierarchical_layers(
    *,
    residual_fn: Callable[[np.ndarray], jnp.ndarray],
    p0_start: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    n_physics: int,
    n_scaling: int,
    n_phi: int,
    mode: str,
    fourier: Any | None,
    l3_lambda: float | None,
    hier_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the L2 hierarchical alternating solve on the inline residual.

    Mirrors ``strategies/heterodyne_hybrid_streaming.py``'s L2 branch: permute
    the native ``[physics | scaling]`` vector to the ``HierarchicalOptimizer``'s
    ``[scaling | physics]`` convention, solve, then un-permute. L3 (when
    ``l3_lambda`` is not None) enters the scalar loss as an SSE-scale per-angle-CV
    penalty (``lambda * (c_CV² + o_CV²) * SSR_data`` over the *reconstructed*
    per-angle scaling — correct for both individual and fourier). It shapes the
    optimizer search only and never contaminates the reported data-only SSR (the
    caller recomputes chi^2 from the data-only residual at the returned popt).

    ``p0_start`` is the BASELINE solution (not the raw seed): alternating
    optimization from the baseline escapes the gradient-cancellation saddle the
    joint solver stalled in, while never moving below it — so the keep-better
    guard accepts deterministically on well-conditioned data and improves on
    degenerate data. ``hier_cfg`` carries the caller's config-derived iteration /
    tolerance budget so ``execute_layers`` users can bound this (otherwise
    expensive) branch.

    Returns ``{"popt", "n_outer", "success"}`` with ``popt`` in native
    ``[physics | scaling]`` layout.
    """
    from xpcsjax.optimization.nlsq.hierarchical import (
        HierarchicalConfig,
        HierarchicalOptimizer,
    )

    hier_cfg = hier_cfg or {}

    # Permute native [physics | scaling] -> hier [scaling | physics].
    perm = np.concatenate(
        [
            np.arange(n_physics, n_physics + n_scaling, dtype=np.intp),
            np.arange(n_physics, dtype=np.intp),
        ]
    )
    unperm = np.empty_like(perm)
    unperm[perm] = np.arange(len(perm), dtype=np.intp)

    p0_hier = np.asarray(p0_start, dtype=np.float64)[perm]
    bounds_hier = (
        np.asarray(lower, dtype=np.float64)[perm],
        np.asarray(upper, dtype=np.float64)[perm],
    )

    def _loss_jax(ph: jnp.ndarray) -> jnp.ndarray:
        params_native = ph[unperm]
        # residual_fn is typed numpy-in; JAX arrays are numpy-compatible at runtime
        # (the JAX/numpy boundary the rest of this module also bridges).
        r = residual_fn(params_native)  # type: ignore[arg-type]
        ssr_data = jnp.sum(r**2)
        if l3_lambda is not None:
            # SSE-scale per-angle-CV penalty over the RECONSTRUCTED per-angle
            # scaling (relative-mode AdaptiveRegularizer equivalent:
            # ``lambda * cv² * mse * n`` summed over the two groups == ``lambda *
            # (c_CV² + o_CV²) * SSR_data``). Correct for both individual and fourier.
            contrasts, offsets = _reconstruct_per_angle_scaling(
                params_native, mode=mode, n_physics=n_physics, n_phi=n_phi, fourier=fourier
            )
            c_cv, o_cv = _per_angle_cv(contrasts, offsets)
            return ssr_data + l3_lambda * (c_cv**2 + o_cv**2) * ssr_data
        return ssr_data

    # JIT both the loss and value-and-grad so HierarchicalOptimizer's many inner
    # evaluations do NOT re-trace the (large) residual graph eagerly on every call
    # — the un-jitted loss made this branch ~100x slower and impractical at ≥1M.
    _loss_jit = jax.jit(_loss_jax)
    _value_and_grad = jax.jit(jax.value_and_grad(_loss_jax))

    def _loss(ph_np: np.ndarray) -> float:
        return float(_loss_jit(jnp.asarray(ph_np)))

    def _grad(ph_np: np.ndarray) -> np.ndarray:
        _val, g = _value_and_grad(jnp.asarray(ph_np))
        return np.asarray(g, dtype=np.float64)

    hier_config = HierarchicalConfig(
        enable=True,
        max_outer_iterations=int(hier_cfg.get("max_outer_iterations", 5)),
        outer_tolerance=float(hier_cfg.get("outer_tolerance", 1e-6)),
        physical_max_iterations=int(hier_cfg.get("physical_max_iterations", 100)),
        per_angle_max_iterations=int(hier_cfg.get("per_angle_max_iterations", 50)),
    )
    optimizer = HierarchicalOptimizer(
        config=hier_config,
        n_phi=n_phi,
        n_physical=n_physics,
        fourier_reparameterizer=fourier if mode == "fourier" else None,
    )
    hier_result = optimizer.fit(
        loss_fn=_loss,
        grad_fn=_grad,
        p0=np.asarray(p0_hier, dtype=np.float64),
        bounds=bounds_hier,
        outer_iteration_callback=None,  # no shear update for heterodyne
    )
    popt_native = np.asarray(hier_result.x, dtype=np.float64)[unperm]
    return {
        "popt": popt_native,
        "n_outer": int(hier_result.n_outer_iterations),
        "success": bool(hier_result.success),
    }


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
    anti_degeneracy_dict: dict | None = None,
) -> Any:
    """Mode-aware heterodyne stratified-LS solve. Returns OptimizationResult.

    Resolves the effective per-angle mode (``averaged`` / ``fourier`` /
    ``individual``) via :func:`_resolve_effective_mode`, computes the
    mode-appropriate scaling-tail seed from per-angle quantiles, and runs a
    single joint pointwise least-squares solve. The objective equals the
    in-memory joint fit for the same mode; the only behavioral change is the
    optional seed-42 reorder/shuffle of the flat point support
    (objective-invariant — reordering residual elements does not change the sum
    of squares).

    The JOINT modes ``averaged``, ``fourier``, and ``individual`` are all
    supported here — objective-consistent with the in-memory
    ``_fit_joint_multi_phi`` path (explicit ``individual`` is a JOINT fit, not
    sequential; ``_aggregate_individual_results`` is only the ``config is
    None``/single-angle fallback).  ``constant`` (frozen scaling) raises
    ``NotImplementedError``; the dispatch gate in ``__init__.py`` only routes
    averaged/fourier/individual here and additionally wraps this driver in a
    best-effort try/except that falls through to the in-memory joint fit.

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
    anti_degeneracy_dict :
        Raw nested ``anti_degeneracy`` YAML block, threaded from dispatch for
        laminar-parity controller banners; ``None`` disables the banner
        side-effect.
    """
    from xpcsjax.optimization.nlsq import heterodyne_logging as _hlog
    from xpcsjax.optimization.nlsq.heterodyne_adapter import NLSQAdapter
    from xpcsjax.optimization.nlsq.heterodyne_core import _resolve_effective_mode
    from xpcsjax.optimization.nlsq.heterodyne_data_prep import far_lag_noise_variance
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
    # __init__.py): ``averaged``, ``fourier``, and ``individual`` all use the
    # JOINT stratified-LS objective, consistent with the in-memory
    # ``_fit_joint_multi_phi`` path (explicit ``individual`` is a joint fit;
    # ``_aggregate_individual_results`` is only the config-is-None /
    # single-angle fallback and never resolves here).
    # ``constant`` freezes scaling and always uses the in-memory path, so the
    # driver refuses to run it even if called directly.
    if mode not in ("averaged", "fourier", "individual"):
        raise NotImplementedError(
            f"stratified-LS supports per_angle_mode in "
            f"('averaged', 'fourier', 'individual'); "
            f"got resolved mode={mode!r} "
            "(constant freezes scaling — use the in-memory joint path)"
        )

    # Laminar-parity narration: announce the path + physical parameter block
    # before the (multi-minute) solve so the two_component log is not silent.
    n_physics_pre = int(model.param_manager.n_varying)
    _hlog.log_stratified_path_activated(int(np.asarray(c2).size))
    _hlog.log_physical_parameters("two_component", list(model.param_manager.varying_names))
    # Laminar-parity: instantiate the shared controller so the heterodyne ≥1M
    # stratified-LS log gains the same Layer 2/3/4 + mode banners laminar emits.
    # Purely for banner/diagnostic-surface parity — numerics below are unchanged.
    # The flat ``hierarchical_active`` / ``regularization_active`` markers stay
    # ``False`` (nothing is executed here); the controller's ``get_diagnostics()``
    # is captured and threaded into ``info["anti_degeneracy"]`` so the public
    # result surfaces the same ``controller_diagnostics`` key laminar emits on
    # its stratified-LS path.  ``None`` on best-effort failure → no key emitted.
    ad_controller = _emit_anti_degeneracy_parity_banners(
        anti_degeneracy_dict=anti_degeneracy_dict,
        phi_deg=np.asarray(phi),
        n_physical=n_physics_pre,
    )

    contrast_pa, offset_pa = compute_quantile_per_angle_scaling(strat)
    contrast_pa = np.asarray(contrast_pa, dtype=np.float64)
    offset_pa = np.asarray(offset_pa, dtype=np.float64)
    _hlog.log_quantile_scaling(contrast_pa, offset_pa)

    fourier: Any | None = None
    if mode == "averaged":
        init_scaling = np.array(
            [float(np.nanmean(contrast_pa)), float(np.nanmean(offset_pa))],
            dtype=np.float64,
        )
        scaling_names = ["contrast", "offset"]
    elif mode == "individual":
        # Per-angle seed: contrast block then offset block, matching
        # make_scaling_expander("individual")'s layout (s[:n_phi], s[n_phi:2*n_phi]).
        init_scaling = np.concatenate([contrast_pa, offset_pa]).astype(np.float64)
        scaling_names = (
            [f"contrast_angle_{i}" for i in range(n_phi)]
            + [f"offset_angle_{i}" for i in range(n_phi)]
        )
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

    _hlog.log_effective_mode(
        mode,
        n_phi=n_phi,
        n_physics=n_physics_pre,
        n_scaling=len(init_scaling),
        threshold=int(getattr(config, "constant_scaling_threshold", 3)),
    )

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
    if len(perm) != x_data0.shape[0]:
        raise RuntimeError(
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
    if len(perm) != x_data.shape[0]:
        raise RuntimeError(
            f"stratification perm length ({len(perm)}) != rebuilt support length "
            f"({x_data.shape[0]}); native and permuted builds disagree on support size"
        )

    n_scaling = int(meta["n_scaling"])
    lower_phys, upper_phys = model.param_manager.get_bounds()
    if mode == "fourier":
        # Fourier coefficients are bounded per the reparameterizer (matches the
        # in-memory _fit_joint_multi_phi path, which uses fourier.get_bounds()).
        if fourier is None:  # invariant: fourier mode always carries a reparameterizer
            raise RuntimeError("fourier per_angle_mode requires a reparameterizer, got None")
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

    # Gradient sanity check (laminar-parity, pre-solve diagnostic). Mirrors the
    # homodyne/laminar ``fit_with_stratified_least_squares`` block: perturb the
    # first PHYSICAL parameter by 1% and verify the summed residual delta is
    # non-negligible, catching a dead-gradient init before the multi-minute solve.
    # Heterodyne's joint vector is PHYSICS-FIRST (``[physics | scaling]``), so the
    # first physical parameter is at index 0 -- NOT laminar's scaling-first index
    # ``2 * n_phi``. Strictly diagnostic: a degenerate gradient raises (mirroring
    # laminar) so the fit fails loudly rather than burning a no-op solve; any other
    # residual-eval error is downgraded to a warning and the solve proceeds. Cost
    # is two evals of the (otherwise-needed) compiled residual function.
    _grad_threshold = 1e-10
    n_physics = len(model.param_manager.varying_names)
    try:
        residuals_0 = np.asarray(residual_fn(p0_full), dtype=np.float64)
        phys_idx = 0  # physics-first layout: first physical parameter
        params_test = np.array(p0_full, dtype=np.float64, copy=True)
        if n_physics > 0 and params_test[phys_idx] != 0.0:
            params_test[phys_idx] *= 1.01  # 1% perturbation
        else:
            # First physical param is exactly 0 (or no physics block) -- additive
            # fallback so the perturbation is never a silent no-op.
            params_test[phys_idx] += 0.01
        residuals_1 = np.asarray(residual_fn(params_test), dtype=np.float64)
        gradient_estimate = float(np.abs(np.sum(residuals_1 - residuals_0)))
        _grad_passed = gradient_estimate >= _grad_threshold
        _hlog.log_gradient_sanity_check(
            residuals_0=residuals_0,
            gradient_estimate=gradient_estimate,
            phys_idx=phys_idx,
            passed=_grad_passed,
            n_params=int(p0_full.size),
            n_physics=n_physics,
            n_scaling=n_scaling,
            threshold=_grad_threshold,
        )
        if not _grad_passed:
            raise ValueError(
                f"Gradient sanity check FAILED: gradient ~{gradient_estimate:.2e} "
                f"(expected > {_grad_threshold:.0e}). Optimization cannot proceed "
                "with zero gradients."
            )
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        if "Gradient sanity check FAILED" in str(exc):
            raise  # Re-raise our custom error -- a degenerate init must fail loudly.
        from xpcsjax.utils.logging import get_logger

        _glog = get_logger(__name__)
        _glog.warning("Gradient sanity check encountered error: %s", exc)
        _glog.warning("Proceeding with optimization, but this may fail")

    _hlog.log_fit_start(int(p0_full.size), int(meta["n_data_points"]), n_chunks=len(chunk_sizes))
    fit = adapter.fit(
        # residual_fn returns a jnp Array; NLSQAdapter types its residual as
        # numpy-returning. JAX arrays are numpy-compatible at runtime, so this is
        # a typing-only impedance at the JAX/numpy boundary.
        residual_fn=residual_fn,  # type: ignore[arg-type]
        initial_params=p0_full,
        bounds=(lower, upper),
        config=config,
    )

    popt = np.asarray(fit.parameters, dtype=np.float64)

    # Post-solve bounds clip (parity with laminar strategies/stratified_ls.py).
    # ``trf`` already respects bounds, so in the normal case popt is already
    # in-bounds and this block is a no-op (ssr/chi2/popt byte-identical).
    # When the solver returns a marginally out-of-bounds value (floating-point
    # edge at a boundary), clipping it in is the correct behaviour and must
    # happen BEFORE the residual recompute so that ssr/chi2/covariance are all
    # derived from the clipped vector.
    _bounds_log = None  # lazy-init — avoids import cost in the normal (no-op) path
    _bounds_violated = False
    for _i in range(popt.size):
        _orig = float(popt[_i])
        if _orig < float(lower[_i]) or _orig > float(upper[_i]):
            popt[_i] = np.clip(_orig, float(lower[_i]), float(upper[_i]))
            _bounds_violated = True
            if _bounds_log is None:
                from xpcsjax.utils.logging import get_logger as _get_logger

                _bounds_log = _get_logger(__name__)
            _pname = (
                joint_param_names[_i]
                if _i < len(joint_param_names)
                else f"param_{_i}"
            )
            _bounds_log.warning(
                "Parameter '%s' violated bounds: %.6e not in [%.6e, %.6e]",
                _pname,
                _orig,
                float(lower[_i]),
                float(upper[_i]),
            )
            _bounds_log.warning("    Clipped to: %.6e (bounds enforced)", float(popt[_i]))
    if _bounds_violated:
        if _bounds_log is None:
            from xpcsjax.utils.logging import get_logger as _get_logger

            _bounds_log = _get_logger(__name__)
        _bounds_log.warning("=" * 80)
        _bounds_log.warning("BOUNDS VIOLATION DETECTED")
        _bounds_log.warning("=" * 80)
        _bounds_log.warning("One or more parameters violated physical bounds.")
        _bounds_log.warning("Parameters have been clipped to valid ranges.")

    # ------------------------------------------------------------------
    # Gated L2/L3 numeric execution (Phase 3, ``execute_layers``).
    # ------------------------------------------------------------------
    # Default OFF: the single baseline solve above IS the result and the honest
    # ``hierarchical_active`` / ``regularization_active`` markers stay False
    # (byte-identical to the pre-Phase-3 path). When ``execute_layers`` is True
    # AND a layer is configured (``enable_hierarchical`` for L2, or
    # ``regularization_mode != "none"`` for L3 — ``regularization.enable`` is
    # IGNORED by the heterodyne config), the layer runs on the SEED (not the
    # baseline) and its result is kept only if the data-only SSR is no worse than
    # the baseline by more than ``tol`` (keep-better). Penalty rows shape the
    # optimizer search only — the reported chi^2 is recomputed from the data-only
    # residual below, so the objective is never contaminated.
    hierarchical_active = False
    regularization_active = False
    _cov_placeholder = False
    _invalidate_adapter_cov = False
    # Metadata for an accepted layer candidate so the reported convergence /
    # iterations / status reflect the LAYER that produced popt, not the stale
    # baseline adapter fit (Fix 3). None on the default / rejected / flag-off path.
    _layer_outcome: dict[str, Any] | None = None
    execute_layers_on = bool(getattr(config, "execute_layers", False))
    execute_layers_status = "off"

    if execute_layers_on:
        from xpcsjax.utils.logging import get_logger as _get_el_logger

        _el_log = _get_el_logger(__name__)
        _keep_tol = 1e-3
        ssr_baseline = float(np.sum(np.asarray(residual_fn(popt), dtype=np.float64) ** 2))
        reg_mode = str(getattr(config, "regularization_mode", "none"))
        l3_configured = (
            (reg_mode != "none")
            and (n_scaling > 0)
            and (mode in ("averaged", "individual", "fourier"))
        )
        enable_hier = bool(getattr(config, "enable_hierarchical", False))
        use_constant = mode == "averaged"

        # L3 rides inside the L2 scalar loss (individual / fourier) as an SSE-scale
        # per-angle-CV penalty; ``None`` disables it. The L3-only branch below uses
        # the same per-angle reconstruction as row-append penalties; averaged L3 is
        # degenerate-zero.
        _l3_lambda = (
            float(getattr(config, "group_variance_lambda", 0.01))
            if (l3_configured and not use_constant)
            else None
        )
        # Config-derived iteration / tolerance budget for the (expensive)
        # hierarchical escape (Fix 2). At ≥1 M points each inner pass materialises
        # the full prediction, so the escape is costly (measured ~3.5x baseline
        # wall-time on C044). Two affordability levers, both safe because the
        # keep-better guard protects quality (fewer iterations only risk a
        # *rejected* — never a *worse* — escape):
        #   * outer: ``hierarchical_max_outer_iterations`` defaults to 20 on the
        #     heterodyne config (tuned for the in-memory path); 20 full-≥1M outer
        #     passes are prohibitive, so the escape CAPS it at ``_ESCAPE_MAX_OUTER``.
        #     Lower it further in config for an even cheaper escape.
        #   * inner: the per-angle / physical inner caps fall back to low defaults
        #     (cheaper than the in-memory 100/50) but honour the config fields if
        #     present (forward-compatible with the laminar controller path).
        _ESCAPE_MAX_OUTER = 5
        _hier_cfg = {
            "max_outer_iterations": min(
                int(getattr(config, "hierarchical_max_outer_iterations", _ESCAPE_MAX_OUTER)),
                _ESCAPE_MAX_OUTER,
            ),
            "outer_tolerance": float(getattr(config, "hierarchical_outer_tolerance", 1e-6)),
            "physical_max_iterations": int(
                getattr(config, "hierarchical_physical_max_iterations", 30)
            ),
            "per_angle_max_iterations": int(
                getattr(config, "hierarchical_per_angle_max_iterations", 20)
            ),
        }

        if enable_hier and n_scaling > 0 and not use_constant:
            try:
                candidate = _run_hierarchical_layers(
                    residual_fn=residual_fn,
                    p0_start=popt,  # refine/escape from the baseline, not the raw seed
                    lower=lower,
                    upper=upper,
                    n_physics=n_physics,
                    n_scaling=n_scaling,
                    n_phi=n_phi,
                    mode=mode,
                    fourier=fourier,
                    l3_lambda=_l3_lambda,
                    hier_cfg=_hier_cfg,
                )
                cand_popt = np.clip(
                    np.asarray(candidate["popt"], dtype=np.float64), lower, upper
                )
                cand_ssr = float(
                    np.sum(np.asarray(residual_fn(cand_popt), dtype=np.float64) ** 2)
                )
                if cand_ssr <= ssr_baseline * (1.0 + _keep_tol):
                    popt = cand_popt
                    hierarchical_active = True
                    regularization_active = bool(l3_configured)
                    _cov_placeholder = True
                    _layer_outcome = {
                        "kind": "L2_hierarchical",
                        "n_outer": int(candidate["n_outer"]),
                        "success": bool(candidate["success"]),
                    }
                    # Honest status: a kept-but-non-converged hierarchical run (SSR
                    # within tolerance yet solver did not meet outer_tolerance) is
                    # surfaced distinctly rather than silently labelled "executed".
                    execute_layers_status = (
                        "executed"
                        if candidate["success"]
                        else "executed_not_converged"
                    )
                    _el_log.info(
                        "execute_layers: L2 hierarchical ACCEPTED "
                        "(SSR %.6e <= baseline %.6e * (1 + %.0e), converged=%s, n_outer=%d).",
                        cand_ssr,
                        ssr_baseline,
                        _keep_tol,
                        bool(candidate["success"]),
                        int(candidate["n_outer"]),
                    )
                else:
                    execute_layers_status = "attempted_but_rejected"
                    _el_log.warning(
                        "execute_layers: L2 hierarchical REJECTED "
                        "(SSR %.6e > baseline %.6e * (1 + %.0e)); keeping baseline.",
                        cand_ssr,
                        ssr_baseline,
                        _keep_tol,
                    )
            except Exception as _exc:  # best-effort: a layer failure must never break the fit
                execute_layers_status = "attempted_but_rejected"
                _el_log.warning(
                    "execute_layers: L2 hierarchical raised (%s: %s); keeping baseline.",
                    type(_exc).__name__,
                    _exc,
                )
        elif l3_configured and use_constant:
            # Averaged: each group is a single scalar -> std == 0 -> the penalty
            # rows are identically zero (mirrors heterodyne_core's degenerate-CV
            # averaged path). L3 is configured-and-objectively-inert here: flag it
            # honestly without a wasteful re-solve (the objective is unchanged, so
            # the baseline popt and SSR are already the L3 result).
            regularization_active = True
            execute_layers_status = "executed"
        elif l3_configured and not use_constant and n_scaling > 0:
            # L3-only (L2 disabled): augment the data residual with
            # ``sqrt(lambda) * CV`` penalty rows and re-solve — the row-append L3
            # of heterodyne_core's fourier path (2451-2515 / plan step 8). The
            # penalty rows shape the least-squares search only; the reported chi^2
            # is recomputed from the data-only residual below, so the objective is
            # never contaminated. The baseline adapter covariance is invalidated
            # (popt moved) so covariance is recomputed (host jacfwd, data-only) at
            # the new popt.
            try:
                _lambda = float(getattr(config, "group_variance_lambda", 0.01))
                _sqrt_lambda = float(np.sqrt(max(_lambda, 0.0)))

                def _l3_augmented_residual(x: np.ndarray) -> jnp.ndarray:
                    r = residual_fn(x)
                    contrasts, offsets = _reconstruct_per_angle_scaling(
                        jnp.asarray(x),
                        mode=mode,
                        n_physics=n_physics,
                        n_phi=n_phi,
                        fourier=fourier,
                    )
                    c_cv, o_cv = _per_angle_cv(contrasts, offsets)
                    penalty_rows = jnp.array(
                        [_sqrt_lambda * c_cv, _sqrt_lambda * o_cv], dtype=jnp.float64
                    )
                    return jnp.concatenate([r, penalty_rows])

                fit_l3 = adapter.fit(
                    residual_fn=_l3_augmented_residual,  # type: ignore[arg-type]
                    initial_params=popt,  # refine from the baseline, not the raw seed
                    bounds=(lower, upper),
                    config=config,
                )
                cand_popt = np.clip(
                    np.asarray(fit_l3.parameters, dtype=np.float64), lower, upper
                )
                cand_ssr = float(
                    np.sum(np.asarray(residual_fn(cand_popt), dtype=np.float64) ** 2)
                )
                if cand_ssr <= ssr_baseline * (1.0 + _keep_tol):
                    popt = cand_popt
                    regularization_active = True
                    _invalidate_adapter_cov = True  # popt moved; baseline cov stale
                    _layer_outcome = {
                        "kind": "L3_row_append",
                        "n_outer": int(getattr(fit_l3, "n_iterations", 0) or 0),
                        "success": bool(getattr(fit_l3, "success", True)),
                    }
                    execute_layers_status = (
                        "executed"
                        if _layer_outcome["success"]
                        else "executed_not_converged"
                    )
                    _el_log.info(
                        "execute_layers: L3 row-append ACCEPTED "
                        "(SSR %.6e <= baseline %.6e * (1 + %.0e), converged=%s).",
                        cand_ssr,
                        ssr_baseline,
                        _keep_tol,
                        _layer_outcome["success"],
                    )
                else:
                    execute_layers_status = "attempted_but_rejected"
                    _el_log.warning(
                        "execute_layers: L3 row-append REJECTED "
                        "(SSR %.6e > baseline %.6e * (1 + %.0e)); keeping baseline.",
                        cand_ssr,
                        ssr_baseline,
                        _keep_tol,
                    )
            except Exception as _exc:  # best-effort: a layer failure must never break the fit
                execute_layers_status = "attempted_but_rejected"
                _el_log.warning(
                    "execute_layers: L3 row-append raised (%s: %s); keeping baseline.",
                    type(_exc).__name__,
                    _exc,
                )
        else:
            execute_layers_status = "no_layers_configured"

    # The baseline adapter covariance is valid only when popt is still the
    # baseline solve. The L3-only row-append branch moves popt, so it invalidates
    # the adapter covariance and forces the host-jacfwd (data-only) recompute at
    # the new popt. The L2 branch uses the identity placeholder instead.
    _pcov_from_adapter = (
        np.asarray(fit.covariance, dtype=np.float64)
        if (fit.covariance is not None and not _invalidate_adapter_cov)
        else None
    )

    # SSR conservation: recompute the data-only residual at the solution and
    # decompose chi^2 by phi index. Mirrors the joint averaged path, which
    # reports ``chi_squared = sum(data_only_residual**2)`` rather than the
    # optimizer's robust-loss cost. We pass ``info["cost"] = 0.5 * SSR`` so the
    # builder's ``chi_squared = info["cost"] * 2`` recovers the exact SSR.
    final_residual = np.asarray(residual_fn(popt), dtype=np.float64)
    ssr = float(np.sum(final_residual**2))

    # Covariance: use the adapter's covariance when available; otherwise compute
    # a host-side Jacobian covariance mirroring laminar's stratified-LS path
    # (strategies/stratified_ls.py lines ~691-710).  At ≥1 M points jacfwd
    # materialises a large (N × n_params) Jacobian, so every known failure mode
    # is caught and falls back to all-NaN (best-effort: a covariance failure must
    # never break the fit).
    if _cov_placeholder:
        # Accepted L2 hierarchical branch: the alternating solve does not produce
        # a Gauss-Newton covariance, so use an identity placeholder (mirrors
        # strategies/heterodyne_hybrid_streaming.py's ``covariance_is_placeholder``).
        pcov: np.ndarray = np.eye(int(popt.size), dtype=np.float64)
    elif _pcov_from_adapter is not None:
        pcov = _pcov_from_adapter
    else:
        n_params = int(popt.size)
        n_data = int(meta["n_data_points"])
        s2 = ssr / max(n_data - n_params, 1)
        try:
            from xpcsjax.utils.logging import get_logger as _get_logger

            _cov_log = _get_logger(__name__)
            _cov_log.info(
                "Computing host Jacobian covariance (s²=%.6e, n_data=%d, n_params=%d).",
                s2,
                n_data,
                n_params,
            )
            # Column-blocked forward-mode Jacobian: byte-identical to
            # jax.jacfwd(residual_fn)(popt) but caps the live AD-tangent width at
            # _COV_JACFWD_COL_BLOCK instead of n_params, cutting the dominant
            # post-solve memory spike at >=1M points (see _chunked_jacfwd_dense).
            J = _chunked_jacfwd_dense(residual_fn, popt)
            JTJ = J.T @ J
            try:
                pcov = np.linalg.inv(JTJ) * s2
            except np.linalg.LinAlgError:
                _cov_log.warning(
                    "Singular Jacobian in heterodyne stratified-LS covariance; "
                    "falling back to pseudo-inverse."
                )
                pcov = np.linalg.pinv(JTJ) * s2
        except (MemoryError, np.linalg.LinAlgError, ValueError, RuntimeError) as _exc:
            from xpcsjax.utils.logging import get_logger as _get_logger

            _get_logger(__name__).warning(
                "Jacobian covariance computation failed (%s: %s); "
                "returning all-NaN covariance matrix.",
                type(_exc).__name__,
                _exc,
            )
            pcov = np.full((n_params, n_params), np.nan)

    phi_idx_flat = np.asarray(x_data[:, 0], dtype=np.int64)
    n_phi_meta = int(meta["n_phi"])
    chi2_per_angle = np.zeros(n_phi_meta, dtype=np.float64)
    np.add.at(chi2_per_angle, phi_idx_flat, final_residual**2)

    # Effective fit-outcome fields (Fix 3): when an execute_layers candidate was
    # accepted, popt came from the LAYER, so success / iterations / message must
    # describe that layer — not the stale baseline adapter fit. Otherwise these are
    # the baseline adapter's real outcome.
    _eff_message: str | None
    if _layer_outcome is not None:
        _eff_success = bool(_layer_outcome["success"])
        _eff_nit = int(_layer_outcome["n_outer"])
        _eff_message = (
            f"{_layer_outcome['kind']} accepted by keep-better guard "
            f"(converged={_layer_outcome['success']})"
        )
        _eff_reason = str(_layer_outcome["kind"])
        _eff_fevals = None
    else:
        _eff_success = bool(fit.success)
        _eff_nit = int(fit.n_iterations or 0)
        _eff_message = getattr(fit, "message", None)
        _eff_reason = str(getattr(fit, "convergence_reason", "") or "")
        _eff_fevals = getattr(fit, "n_function_evals", None)
    _eff_wall = float(fit.wall_time_seconds or 0.0)

    # Laminar-parity OPTIMIZATION RESULTS block (reads the effective fit outcome).
    # ``initial_cost`` is omitted (None) to keep the stratified-LS path at ZERO
    # extra residual evaluations — the block then reports the final cost without a
    # cost-reduction percentage.
    _n_data = int(meta["n_data_points"])
    # Noise-normalized reduced chi^2 (targets ~1.0), mirroring the in-memory
    # averaged/fourier joint paths. Raw SSR/dof collapses to MSE << 1 on
    # normalized C2 data and is not an interpretable goodness-of-fit. The far-lag
    # photon-noise variance is threaded to the result builder via ``sigma2_noise``
    # so the logged value and the OptimizationResult agree.
    _sigma2_noise = far_lag_noise_variance(c2)
    _n_dof = max(1, _n_data - int(popt.size))
    _reduced_chi2 = ssr / (_sigma2_noise * _n_dof) if _sigma2_noise > 1e-12 else ssr / _n_dof
    _hlog.log_optimization_results(
        success=_eff_success,
        message=_eff_message,
        n_iterations=_eff_nit,
        initial_cost=None,
        final_cost=0.5 * ssr,
        wall_time=_eff_wall,
        function_evals=_eff_fevals,
    )

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
    _hlog.log_stratification_diagnostics(
        strat_diag, n_chunks=len(chunk_sizes), n_points=_n_data, n_phi=n_phi_meta
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

    _hlog.log_stratified_complete(ssr, _reduced_chi2)

    info = {
        # Effective outcome (Fix 3): reflects the accepted layer when one replaced
        # popt, else the baseline adapter fit.
        "success": _eff_success,
        # SciPy termination reason (status->reason string from build_result_from_nlsq),
        # or the layer kind when a layer was accepted. Lets the result builder report
        # ``max_iter`` (graded on chi^2) instead of a blanket ``failed``/``poor`` when
        # the solver merely exhausted ``max_nfev``.
        "convergence_reason": _eff_reason,
        "cost": 0.5 * ssr,
        "nit": _eff_nit,
        "wall_time": _eff_wall,
        "n_data_points": int(meta["n_data_points"]),
        "sigma2_noise": float(_sigma2_noise),
        "stratification_memory": mem_estimate,
    }
    # Thread the anti-degeneracy block into info so build_hybrid_streaming_result
    # surfaces the activation markers + (when present) ``controller_diagnostics``
    # in the public nlsq_diagnostics block, reaching parity with laminar's
    # stratified-LS path (strategies/stratified_ls.py line ~862).
    #
    # ``hierarchical_active`` / ``regularization_active`` are read by the result
    # builder via ``ad_block.get(..., False)`` — they are HONEST per the executed
    # branch: ``False`` on the default (flag-off) single-solve path, and ``True``
    # only when the gated ``execute_layers`` L2/L3 candidate was accepted by the
    # keep-better guard above. ``execute_layers_status`` /
    # ``covariance_is_placeholder`` are surfaced only when the flag is on (so the
    # flag-off result surface stays byte-identical). (shear_weighting is L5-gated
    # off for two_component and set to the heterodyne sentinel inside the builder.)
    _ad_block: dict[str, Any] = {
        "hierarchical_active": hierarchical_active,
        "regularization_active": regularization_active,
        "per_angle_mode": mode,
    }
    if execute_layers_on:
        _ad_block["execute_layers"] = True
        _ad_block["execute_layers_status"] = execute_layers_status
        if _layer_outcome is not None:
            # Surface the accepted-layer provenance so production triage sees what
            # actually produced popt (Fix 3).
            _ad_block["execute_layers_kind"] = _layer_outcome["kind"]
            _ad_block["execute_layers_n_outer"] = int(_layer_outcome["n_outer"])
            _ad_block["execute_layers_converged"] = bool(_layer_outcome["success"])
    if _cov_placeholder:
        _ad_block["covariance_is_placeholder"] = True
    # Best-effort: controller diagnostics only when the controller was built.
    if ad_controller is not None:
        _ad_block["controller_diagnostics"] = ad_controller.get_diagnostics()
    info["anti_degeneracy"] = _ad_block
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
