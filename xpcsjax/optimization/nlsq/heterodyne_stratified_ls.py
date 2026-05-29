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

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from xpcsjax.optimization.nlsq.strategies.chunking import (
    create_angle_stratified_indices,
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
        If True, apply a fixed-seed permutation after stratification to break
        angle-sequential ordering (homodyne local-minimum-avoidance parity).
    seed : int
        Shuffle seed (fixed at 42 for reproducibility; matches homodyne).

    Returns
    -------
    (perm, chunk_sizes) : tuple[np.ndarray, list[int]]
        ``perm`` reorders any per-point array; ``chunk_sizes`` are the
        interleaved chunk sizes from stratification.
    """
    perm, chunk_sizes = create_angle_stratified_indices(phi_flat, target_chunk_size)
    perm = np.asarray(perm, dtype=np.int64)
    if shuffle:
        rng = np.random.RandomState(seed)
        perm = perm[rng.permutation(len(perm))]
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
    - individual: 2*n_phi params (contrast block then offset block). [Phase 2]
    - fourier: 2*(2K+1) Fourier coefficients via ``fourier``.            [Phase 2]
    """
    if per_angle_mode == "averaged":

        def expand(s: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            return jnp.full((n_phi,), s[0]), jnp.full((n_phi,), s[1])

        return expand, 2

    raise NotImplementedError(
        f"scaling expander for per_angle_mode={per_angle_mode!r} lands in Phase 2"
    )


def build_joint_pointwise_residual(
    *,
    model: Any,
    stratified_data: Any,
    per_angle_mode: str,
    avg_contrast: float,
    avg_offset: float,
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
        One of the modes accepted by ``make_scaling_expander`` (``"averaged"``;
        ``"individual"`` / ``"fourier"`` in Phase 2).
    avg_contrast, avg_offset :
        Initial scalar contrast/offset used to seed the scaling tail of
        ``p0_full`` (truncated to ``n_scaling`` for the averaged case).
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

    if n_scaling != 2:
        raise NotImplementedError(
            f"p0 seeding for per_angle_mode={per_angle_mode!r} (n_scaling={n_scaling}) "
            "lands in Phase 2 (Task 8)"
        )
    init_scaling = np.array([avg_contrast, avg_offset], dtype=np.float64)[:n_scaling]
    p0_full = np.concatenate(
        [
            np.asarray(model.param_manager.get_initial_values(), dtype=np.float64),
            init_scaling,
        ]
    )
    out_meta = {**meta, "n_physics": n_physics, "n_phi": n_phi, "n_scaling": n_scaling}
    return residual_fn, x_data, y_data, p0_full, out_meta
