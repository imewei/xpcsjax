"""Phase 2.3 Step 1 — integration proof: the shared homodyne stratification
engine (``StratifiedResidualFunctionJIT``), fed heterodyne data + a
``HeterodynePointEvaluator`` + the Task-2.2 layout conversion, must reproduce
the heterodyne fit objective (SSR) at a fixed parameter vector, for the three
in-scope per-angle scaling modes (``fixed_constant``, ``individual``,
``auto_averaged``).

This is a DISCOVERY / MEASUREMENT test (not red-green TDD) and TEST-ONLY: it
touches NO production dispatch. It validates the integration approach BEFORE any
production wiring. A mode that does not reconcile is a valid Step-1 finding — do
NOT loosen the assertion or touch production code.

WHAT IS COMPARED (the Step-0 reconciliation — the crux)
-------------------------------------------------------
The engine residual is ``(model - data) / sigma`` over the chunk points, with
the **diagonal (t1 == t2) masked** and padding zeroed. ``sigma`` is uniform
(``build_heterodyne_stratified_data(weights=None)`` -> all ones), so the engine
sums squared raw off-diagonal residuals over the **full** (n_t x n_t) grid.

Two convention gaps had to be reconciled to get an apples-to-apples objective:

1. **sigma weighting.** The engine divides by ``sigma``. We build the stratified
   data with the default ``sigma = 1`` (no weights), so the division is the
   identity and the engine SSR is the plain sum of squared residuals.

2. **diagonal / frame-0 masking.** The engine masks ONLY the diagonal
   (``t1 == t2``); it KEEPS the t-index-0 row/column. The production pointwise
   builder ``build_heterodyne_pointwise_model`` additionally drops the
   t-index-0 boundary (``t1_idx > 0 & t2_idx > 0``), yielding a SMALLER
   ``(n_t-1)*(n_t-2)`` support per angle. Those two supports differ, so the
   builder's pointwise SSR is NOT the engine's objective. We therefore do NOT
   reuse the pointwise SSR as the reference. Instead the reference is computed
   on the engine's OWN support: the heterodyne **meshgrid** kernel
   ``compute_c2_heterodyne`` evaluated per angle, with only the diagonal masked
   (``~eye``) and ``sigma = 1``. This is exactly the surface the engine
   evaluates through ``HeterodynePointEvaluator`` -> ``compute_c2_heterodyne``,
   so the two objectives must agree to machine epsilon when the physics and
   per-angle scaling match.

   We must keep the FULL ``model.t`` grid on the engine side (do not pre-filter
   frame-0 out of the stratified data): the engine derives ``t1_unique`` from
   the data values and passes that grid to the meshgrid kernel, and the
   heterodyne physics integrals depend on the absolute time values. Dropping
   frame-0 would shrink ``t1_unique`` and silently change the physics.

PER-MODE ENGINE CONSTRUCTION (verified against residual_jit.py:304-322)
-----------------------------------------------------------------------
* ``fixed_constant`` -> ``per_angle_scaling=False`` with
  ``fixed_contrast_per_angle = meta["contrast_arr"]`` /
  ``fixed_offset_per_angle = meta["offset_arr"]``. Engine param vector is
  physics-only (n_physics). Layout conversion is the identity.
* ``individual`` -> ``per_angle_scaling=True``. Engine param vector is
  ``[contrast(n_phi) | offset(n_phi) | physics]`` via
  ``physics_first_to_scaling_first(p0, mode="individual", ...)`` (a pure block
  permutation of the physics-first ``p0``).
* ``auto_averaged`` -> ``per_angle_scaling=True`` with the BROADCAST vector
  ``physics_first_to_scaling_first(p0, mode="auto_averaged", ...)`` (the 2
  averaged scalars expanded to ``2*n_phi``; the engine has no compressed
  averaged mode).

The dataset is ``make_synthetic_two_component(n_phi=4, n_t=12)`` with a
non-monotonic angle order, mirroring ``test_pointwise_joint_parity``. ``fourier``
is intentionally out of scope (it is a learned reparameterization that stays on
the heterodyne path, per ``heterodyne_layout.IN_SCOPE_MODES``).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq.heterodyne_layout import (
    IN_SCOPE_MODES,
    physics_first_to_scaling_first,
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

# Non-monotonic angle order (same spirit as test_pointwise_joint_parity): proves
# the engine's sorted-phi searchsorted gather aligns with the caller's order.
_PHI_ORDER = np.array([2, 0, 3, 1])

# The three in-scope conversion modes. ``fourier`` stays on the heterodyne path.
_MODES = ("fixed_constant", "individual", "auto_averaged")


def _effective_scaling_in_phi_unique_order(
    mode: str,
    p0: np.ndarray,
    meta: dict,
    n_phi: int,
    n_varying: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(contrast, offset) per angle in SORTED phi_unique order — exactly the
    scaling the engine resolves at this ``p0`` (mirrors the model_fn slicing and
    ``physics_first_to_scaling_first``)."""
    tail = np.asarray(p0[n_varying:], dtype=np.float64)
    if mode == "fixed_constant":
        return (
            np.asarray(meta["contrast_arr"], dtype=np.float64),
            np.asarray(meta["offset_arr"], dtype=np.float64),
        )
    if mode == "auto_averaged":
        return (
            np.full(n_phi, float(tail[0]), dtype=np.float64),
            np.full(n_phi, float(tail[1]), dtype=np.float64),
        )
    # individual: tail = [contrast(n_phi) | offset(n_phi)]
    return tail[:n_phi].copy(), tail[n_phi:].copy()


def _reference_ssr_on_engine_support(
    *,
    model,
    c2: np.ndarray,
    phi: np.ndarray,
    physics_vec: np.ndarray,
    contrasts: np.ndarray,
    offsets: np.ndarray,
) -> float:
    """SSR over the ENGINE's masked support: the heterodyne meshgrid kernel per
    angle, diagonal masked (``~eye``), sigma = 1 — the apples-to-apples objective.

    ``contrasts`` / ``offsets`` are in sorted phi_unique order; ``c2`` rows are in
    the caller's (possibly non-monotonic) phi order, so we map each phi_unique
    slot back to its row in ``c2`` before differencing.
    """
    import jax.numpy as jnp

    from xpcsjax.core.heterodyne_jax_backend import compute_c2_heterodyne

    phi = np.asarray(phi, dtype=np.float64)
    phi_unique = np.array(sorted(set(phi.tolist())), dtype=np.float64)
    t = np.asarray(model.t, dtype=np.float64)
    n_t = len(t)
    off_diag = ~np.eye(n_t, dtype=bool)

    full = np.asarray(model.param_manager.get_full_values(), dtype=np.float64).copy()
    full[np.asarray(model.param_manager.varying_indices)] = np.asarray(physics_vec)

    ssr = 0.0
    for k, phi_val in enumerate(phi_unique):
        # phi_unique[k] -> row index in the caller-ordered c2 / phi.
        in_idx = int(np.where(phi == phi_val)[0][0])
        grid = np.asarray(
            compute_c2_heterodyne(
                jnp.asarray(full),
                jnp.asarray(t),
                float(model.q),
                float(model.dt),
                float(phi_val),
                float(contrasts[k]),
                float(offsets[k]),
            )
        )
        resid = (grid - c2[in_idx])[off_diag]  # sigma = 1, diagonal excluded
        ssr += float(np.sum(resid**2))
    return ssr


def _build_engine_for_mode(
    *,
    mode: str,
    chunked,
    phys_names: list[str],
    n_varying: int,
    n_phi: int,
    p0: np.ndarray,
    meta: dict,
) -> tuple[StratifiedResidualFunctionJIT, np.ndarray]:
    """Construct the engine + the layout-converted parameter vector for ``mode``.

    Returns ``(engine, engine_param_vector)``.
    """
    evaluator = HeterodynePointEvaluator(
        analysis_mode="two_component",
        q=float(_MODEL_Q),
        dt=float(_MODEL_DT),
    )

    if mode == "fixed_constant":
        engine = StratifiedResidualFunctionJIT(
            stratified_data=chunked,
            per_angle_scaling=False,
            physical_param_names=phys_names,
            fixed_contrast_per_angle=np.asarray(meta["contrast_arr"], dtype=np.float64),
            fixed_offset_per_angle=np.asarray(meta["offset_arr"], dtype=np.float64),
            evaluator=evaluator,
        )
    else:  # individual / auto_averaged -> per-angle (expanded) scaling
        engine = StratifiedResidualFunctionJIT(
            stratified_data=chunked,
            per_angle_scaling=True,
            physical_param_names=phys_names,
            fixed_contrast_per_angle=None,
            fixed_offset_per_angle=None,
            evaluator=evaluator,
        )

    engine_vec = physics_first_to_scaling_first(
        np.asarray(p0, dtype=np.float64),
        n_physics=n_varying,
        mode=mode,
        n_phi=n_phi,
    )
    return engine, engine_vec


# Per-fit constants captured at module import for the evaluator builder. They are
# set the first time ``_make_case`` runs (all cases share the same model config).
_MODEL_Q = 0.0
_MODEL_DT = 0.0


def _make_case():
    """Build the shared synthetic two-component case (non-monotonic phi)."""
    global _MODEL_Q, _MODEL_DT
    model, c2, phi = make_synthetic_two_component(n_phi=4, n_t=12)
    c2, phi = c2[_PHI_ORDER], phi[_PHI_ORDER]
    _MODEL_Q = float(model.q)
    _MODEL_DT = float(model.dt)
    return model, c2, phi


def test_in_scope_modes_are_the_three_under_test():
    """Guard: the modes this proof exercises are exactly the layout-conversion
    in-scope set (so a future change to ``IN_SCOPE_MODES`` surfaces here)."""
    assert set(_MODES) == set(IN_SCOPE_MODES), (
        f"modes under test {set(_MODES)} != IN_SCOPE_MODES {set(IN_SCOPE_MODES)}"
    )


@pytest.mark.parametrize("mode", _MODES)
def test_engine_routes_heterodyne_residual_matches_objective(mode):
    """Engine SSR (heterodyne data + HeterodynePointEvaluator + layout convert)
    == heterodyne meshgrid SSR on the engine's masked support, at a fixed p0."""
    import jax.numpy as jnp

    model, c2, phi = _make_case()
    phys_names = list(model.param_manager.varying_names)
    n_varying = len(phys_names)
    n_phi = len(phi)

    strat = build_heterodyne_stratified_data(model, c2, np.asarray(phi))
    chunked = create_stratified_chunks(strat, target_chunk_size=100_000)

    # build_heterodyne_pointwise_model gives us the canonical physics-first p0
    # (varying physics + the mode's scaling tail) and the frozen quantile scaling
    # in meta — the SAME starting vector a production heterodyne fit would use.
    _model_fn, _x, _y, p0, meta = build_heterodyne_pointwise_model(
        stratified_data=strat,
        model=model,
        physical_param_names=phys_names,
        per_angle_mode=mode,
    )
    p0 = np.asarray(p0, dtype=np.float64)
    physics = p0[:n_varying]

    contrasts, offsets = _effective_scaling_in_phi_unique_order(
        mode, p0, meta, n_phi, n_varying
    )

    # Reference objective on the engine's support (diagonal-masked, sigma=1).
    ref_ssr = _reference_ssr_on_engine_support(
        model=model,
        c2=c2,
        phi=phi,
        physics_vec=physics,
        contrasts=contrasts,
        offsets=offsets,
    )

    engine, engine_vec = _build_engine_for_mode(
        mode=mode,
        chunked=chunked,
        phys_names=phys_names,
        n_varying=n_varying,
        n_phi=n_phi,
        p0=p0,
        meta=meta,
    )

    # Engine param-vector length contract (residual_jit param slicing):
    #   fixed_constant -> physics-only;  individual/auto_averaged -> 2*n_phi + physics.
    expected_len = n_varying if mode == "fixed_constant" else 2 * n_phi + n_varying
    assert engine_vec.shape == (expected_len,), (
        f"mode={mode}: engine vector length {engine_vec.shape} != ({expected_len},)"
    )

    residual = np.asarray(engine(jnp.asarray(engine_vec)), dtype=np.float64)
    engine_ssr = float(np.sum(residual**2))

    rel = abs(engine_ssr - ref_ssr) / max(abs(ref_ssr), 1e-300)
    assert np.isclose(engine_ssr, ref_ssr, rtol=1e-8, atol=0.0), (
        f"mode={mode}: engine SSR {engine_ssr!r} != heterodyne reference SSR "
        f"{ref_ssr!r} (rel_diff={rel:.3e}). The shared engine's residual "
        "convention does not reconcile with the heterodyne objective on this "
        "mode — a Step-1 finding (likely a sigma/masking/ordering mismatch)."
    )


# ===========================================================================
# Phase 2.3a — PRODUCTION-parity proof (frame-0 reconciled)
# ===========================================================================
# Step 1 above compared the engine against a reference computed on the ENGINE's
# OWN masked support (diagonal-only, frame-0 KEPT -> ``n_t*(n_t-1)`` per angle).
# That proves the engine evaluates the correct heterodyne physics, but it is NOT
# the production objective: the production heterodyne residual
# ``compute_multi_angle_residuals`` excludes BOTH the diagonal (``t1==t2``) AND
# the t=0 boundary (``t1_idx==0 OR t2_idx==0``) via ``_offdiag_indices`` ->
# ``(n_t-1)*(n_t-2)`` support per angle. Step 1's larger support means its SSR is
# strictly above production's; the two are not bit-comparable by construction.
#
# THE RECONCILIATION (the whole point of Phase 2.3a)
# --------------------------------------------------
# Build the engine's stratified chunks EXCLUDING frame-0 — keep only pairs with
# both ``t1 > t[0]`` and ``t2 > t[0]`` (``keep = (t1_idx>0) & (t2_idx>0)``) — but
# KEEP the on-grid diagonal (the engine's own ``non_diagonal`` mask zeros it).
# The engine then sees a reduced ``(n_t-1) x (n_t-1)`` time grid; after its
# diagonal masking the contributing support is ``(n_t-1)*(n_t-2)`` per angle —
# EXACTLY the production support. So the engine objective must equal the
# production objective ``sum(compute_multi_angle_residuals(...)**2)`` to machine
# epsilon.
#
# Why dropping frame-0 from the flat data is physics-safe here: the heterodyne
# kernel's transport/velocity terms are ``cumsum(...,dt)`` DIFFERENCES
# (``create_signed_integral_matrix`` -> ``cumsum[j]-cumsum[i]``). Removing the
# frame-0 row/col shifts the cumsum anchor by a constant that cancels in every
# interior ``(i>0, j>0)`` pair difference; verified empirically to ~4e-15 rel.
# (Step 1's docstring note "dropping frame-0 would change the physics" referred
# to a NAIVE float32 pass; under the mandatory ``JAX_ENABLE_X64=1`` the kept-pair
# kernel values are anchor-independent at machine precision.)
#
# If a mode does NOT reconcile against PRODUCTION, that is a real finding — the
# assertion must NOT be loosened and production code must NOT be touched.


def _drop_frame0_stratified_data(
    strat: HeterodyneStratifiedData,
    *,
    t: np.ndarray,
    n_phi: int,
) -> HeterodyneStratifiedData:
    """Return a copy of ``strat`` with every (t1, t2) pair touching frame-0
    (``t1 == t[0]`` OR ``t2 == t[0]``) removed from the flat arrays.

    The on-grid diagonal is KEPT (the engine masks it). The reduced time grid is
    ``t[1:]`` (length ``n_t-1``), so ``sigma`` is rebuilt as ones of shape
    ``(n_phi, n_t-1, n_t-1)`` to stay index-aligned with the engine's
    ``searchsorted`` gather over the shrunken ``t1_unique``/``t2_unique``.
    """
    t = np.asarray(t, dtype=np.float64)
    t0 = float(t[0])
    eps = float(strat.dt) * 1e-6
    keep = (strat.t1_flat > t0 + eps) & (strat.t2_flat > t0 + eps)

    # Rebuild per-angle chunk_sizes from the filtered mask (angle slabs are
    # contiguous in the flat layout, in chunk_sizes order).
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


def _production_reference_ssr(
    *,
    model,
    c2: np.ndarray,
    phi: np.ndarray,
    physics_vec: np.ndarray,
    contrasts: np.ndarray,
    offsets: np.ndarray,
) -> float:
    """PRODUCTION heterodyne objective: ``sum(compute_multi_angle_residuals**2)``.

    ``compute_multi_angle_residuals`` excludes the diagonal AND frame-0 via
    ``_offdiag_indices`` -> support ``(n_t-1)*(n_t-2)`` per angle. ``weights`` are
    ones (so ``sqrt(w) == 1`` and the residual is the raw model-minus-data),
    matching the engine's ``sigma = 1``.

    ``contrasts``/``offsets`` are per-angle ``(n_phi,)`` in sorted phi_unique
    order; the data batch is assembled in that same order so the production
    vmap (which consumes ``phi_angles`` positionally) sees a consistent layout.
    """
    import jax.numpy as jnp

    from xpcsjax.core.heterodyne_jax_backend import compute_multi_angle_residuals

    phi = np.asarray(phi, dtype=np.float64)
    phi_unique = np.array(sorted(set(phi.tolist())), dtype=np.float64)
    t = np.asarray(model.t, dtype=np.float64)

    full = np.asarray(model.param_manager.get_full_values(), dtype=np.float64).copy()
    full[np.asarray(model.param_manager.varying_indices)] = np.asarray(physics_vec)

    # Reorder the (caller-ordered) c2 rows into sorted phi_unique order so that
    # row k corresponds to phi_unique[k] / contrasts[k] / offsets[k].
    rows_for_unique = [int(np.where(phi == pv)[0][0]) for pv in phi_unique]
    c2_batch = np.stack([np.asarray(c2[r], dtype=np.float64) for r in rows_for_unique])
    weights_batch = np.ones_like(c2_batch)

    residuals = np.asarray(
        compute_multi_angle_residuals(
            jnp.asarray(full),
            jnp.asarray(t),
            float(model.q),
            float(model.dt),
            jnp.asarray(phi_unique),
            jnp.asarray(c2_batch),
            jnp.asarray(weights_batch),
            jnp.asarray(np.asarray(contrasts, dtype=np.float64)),
            jnp.asarray(np.asarray(offsets, dtype=np.float64)),
        )
    )
    return float(np.sum(residuals**2))


@pytest.mark.parametrize("mode", _MODES)
def test_engine_route_matches_production_objective_frame0_reconciled(mode):
    """Engine SSR over the FRAME-0-EXCLUDED chunks == the PRODUCTION heterodyne
    objective ``sum(compute_multi_angle_residuals**2)``, at a fixed p0.

    This is the production-parity proof: after dropping frame-0 from the engine
    chunks, the engine's effective support (``(n_t-1)*(n_t-2)`` per angle, after
    its own diagonal masking) equals the production support, so the two
    objectives must agree to machine epsilon.
    """
    import jax.numpy as jnp

    model, c2, phi = _make_case()
    phys_names = list(model.param_manager.varying_names)
    n_varying = len(phys_names)
    n_phi = len(phi)
    t = np.asarray(model.t, dtype=np.float64)
    n_t = len(t)

    # FULL stratified data first (so build_heterodyne_pointwise_model resolves the
    # canonical physics-first p0 + quantile scaling exactly as production would),
    # then drop frame-0 from the engine's chunks.
    strat_full = build_heterodyne_stratified_data(model, c2, np.asarray(phi))
    strat = _drop_frame0_stratified_data(strat_full, t=t, n_phi=n_phi)
    chunked = create_stratified_chunks(strat, target_chunk_size=100_000)

    _model_fn, _x, _y, p0, meta = build_heterodyne_pointwise_model(
        stratified_data=strat_full,
        model=model,
        physical_param_names=phys_names,
        per_angle_mode=mode,
    )
    p0 = np.asarray(p0, dtype=np.float64)
    physics = p0[:n_varying]

    contrasts, offsets = _effective_scaling_in_phi_unique_order(
        mode, p0, meta, n_phi, n_varying
    )

    # PRODUCTION objective (excludes diagonal + frame-0 -> (n_t-1)*(n_t-2)).
    ssr_prod = _production_reference_ssr(
        model=model,
        c2=c2,
        phi=phi,
        physics_vec=physics,
        contrasts=contrasts,
        offsets=offsets,
    )

    engine, engine_vec = _build_engine_for_mode(
        mode=mode,
        chunked=chunked,
        phys_names=phys_names,
        n_varying=n_varying,
        n_phi=n_phi,
        p0=p0,
        meta=meta,
    )

    residual = np.asarray(engine(jnp.asarray(engine_vec)), dtype=np.float64)
    ssr_engine = float(np.sum(residual**2))

    # Guard: frame-0 was ACTUALLY excluded. The engine keeps the reduced-grid
    # diagonal in ``mask`` (real points = (n_t-1)^2 per angle) and zeros it at
    # residual-compute time, so the CONTRIBUTING support is the nonzero residual
    # count = (n_t-1)*(n_t-2) per angle == production. If frame-0 had NOT been
    # excluded these would instead be n_t^2 and n_t*(n_t-1) -> the test would be
    # silently measuring the wrong (larger) support.
    n_mask_real = int(np.sum(engine.mask))
    n_contributing = int(np.sum(residual != 0.0))
    assert n_mask_real == (n_t - 1) ** 2 * n_phi, (
        f"mode={mode}: engine real-point count {n_mask_real} != "
        f"(n_t-1)^2 * n_phi = {(n_t - 1) ** 2 * n_phi} — frame-0 not excluded "
        "from the chunks (the reduced grid should be (n_t-1) x (n_t-1))."
    )
    assert n_contributing == (n_t - 1) * (n_t - 2) * n_phi, (
        f"mode={mode}: contributing support {n_contributing} != "
        f"(n_t-1)*(n_t-2) * n_phi = {(n_t - 1) * (n_t - 2) * n_phi} — does not "
        "match the production (diagonal + frame-0 excluded) support."
    )

    rel = abs(ssr_engine - ssr_prod) / max(abs(ssr_prod), 1e-300)
    assert np.isclose(ssr_engine, ssr_prod, rtol=1e-8, atol=0.0), (
        f"mode={mode}: engine SSR {ssr_engine!r} != PRODUCTION heterodyne SSR "
        f"{ssr_prod!r} (rel_diff={rel:.3e}). Routing two_component through the "
        "shared engine with frame-0 reconciliation does NOT reproduce the "
        "production compute_multi_angle_residuals objective on this mode — a "
        "real Phase 2.3a finding. Do NOT loosen this assertion or fall back to "
        "the engine-support reference; diagnose the mismatch."
    )
