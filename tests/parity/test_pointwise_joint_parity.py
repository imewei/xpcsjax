"""Phase-0 gate: the flat point-wise heterodyne joint residual must reproduce the
batched joint-fit objective (SSR) to rtol=1e-10 for EVERY canonical per-angle
scaling mode, at a fixed parameter vector. Precondition for routing two_component
through the shared homodyne stratification engine.

This is a DISCOVERY / MEASUREMENT test, not red-green TDD. It measures, at a fixed
``p0``, whether the flat point-wise model SSR equals the batched joint residual SSR
when fed the SAME physics params and the SAME effective per-angle scaling. A failure
on a mode is a valid Phase-0 finding (do not loosen the assertion or touch production
code) — the production code under measurement is:

* ``compute_multi_angle_residuals`` (batched joint residual)
* ``build_heterodyne_pointwise_model`` (flat point-wise model + its scaling map)

The three canonical resolved modes are ``constant`` / ``averaged`` / ``individual``.
Phase 4 retired the truncated-basis per-angle mode from the pointwise builder (the
builder now rejects it with ``ValueError`` and no longer accepts the former basis-order
kwarg), so the former truncated-basis cases are gone. The builder
emits the canonical **scaling-first** ``p0`` ``[scaling_head | physics_tail]``:
``averaged`` head ``[c_avg, o_avg]``, ``individual`` head
``[contrast(n_phi) | offset(n_phi)]``, ``constant`` head empty.
"""

import numpy as np
import pytest

from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.core.heterodyne_jax_backend import compute_multi_angle_residuals
from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
    build_heterodyne_stratified_data,
)
from xpcsjax.optimization.nlsq.strategies.heterodyne_hybrid_streaming import (
    build_heterodyne_pointwise_model,
)


def _effective_scaling(mode, p0, meta, n_phi, n_varying):
    """(contrast_per_angle, offset_per_angle) in SORTED phi_unique order, matching
    EXACTLY what build_heterodyne_pointwise_model's model_fn computes at p0.

    The builder emits the canonical **scaling-first** ``p0``
    ``[scaling_head | physics_tail]`` — the scaling head is at the FRONT, so the
    head slice is ``p0[:n_head]`` (NOT a trailing tail).

    Reconstructed against the REAL APIs (verified against source):

    * ``constant``   — frozen quantile scaling lives in ``meta`` (head empty).
    * ``averaged``   — head = [contrast_scalar, offset_scalar], broadcast.
    * ``individual`` — head = [contrast(n_phi) | offset(n_phi)].
    """
    if mode == "constant":
        return (
            np.asarray(meta["contrast_arr"], float),
            np.asarray(meta["offset_arr"], float),
        )
    head = np.asarray(p0[: len(p0) - n_varying], dtype=float)
    if mode == "averaged":
        return np.full(n_phi, float(head[0])), np.full(n_phi, float(head[1]))
    if mode == "individual":
        return head[:n_phi].copy(), head[n_phi:].copy()
    raise AssertionError(mode)


def _assert_pointwise_matches_batched(model, c2, phi, mode) -> dict:
    """Run the full point-wise vs batched SSR comparison for one mode at p0.

    Steps (the shared ~7-step plumbing both tests need):
        1. build stratified data
        2. build_heterodyne_pointwise_model
        3. flat point-wise SSR at p0
        4. _effective_scaling -> per-angle (contrast, offset) in phi_unique order
        5. searchsorted reindex from phi_unique order back to caller's phi order
        6. batched compute_multi_angle_residuals SSR at the SAME physics + scaling
        7. assert SSR parity at rtol=1e-10

    Returns ``meta`` so callers can apply extra mode-specific guards inline.
    """
    import jax.numpy as jnp

    strat = build_heterodyne_stratified_data(model, c2, np.asarray(phi))
    phys_names = list(model.param_manager.varying_names)
    n_varying = len(phys_names)

    model_fn, x_data, y_data, p0, meta = build_heterodyne_pointwise_model(
        stratified_data=strat,
        model=model,
        physical_param_names=phys_names,
        per_angle_mode=mode,
    )

    pw = np.asarray(model_fn(jnp.asarray(x_data), *p0))
    pw_ssr = float(np.sum((pw - np.asarray(y_data)) ** 2))

    c_sorted, o_sorted = _effective_scaling(mode, p0, meta, len(phi), n_varying)
    phi_unique = np.asarray(meta["phi_unique"], float)
    pos = np.searchsorted(phi_unique, np.asarray(phi, float))
    contrasts, offsets = c_sorted[pos], o_sorted[pos]

    # p0 is scaling-first: physics is the TAIL slice ``p0[-n_varying:]``.
    full = np.asarray(model.param_manager.get_full_values(), dtype=float).copy()
    full[np.asarray(model.param_manager.varying_indices)] = np.asarray(p0[-n_varying:])

    weights = jnp.ones_like(jnp.asarray(c2))
    r = compute_multi_angle_residuals(
        jnp.asarray(full),
        model.t,
        model.q,
        model.dt,
        jnp.asarray(phi),
        jnp.asarray(c2),
        weights,
        jnp.asarray(contrasts),
        jnp.asarray(offsets),
    )
    batched_ssr = float(jnp.sum(jnp.asarray(r) ** 2))

    assert np.isclose(pw_ssr, batched_ssr, rtol=1e-10, atol=0.0), (
        f"mode={mode}: pointwise SSR {pw_ssr!r} != batched SSR {batched_ssr!r}"
    )
    return meta


@pytest.mark.parametrize("mode", ["constant", "averaged", "individual"])
def test_pointwise_joint_ssr_matches_batched(mode):
    model, c2, phi = make_synthetic_two_component(n_phi=4, n_t=16)
    order = np.array([2, 0, 3, 1])  # non-monotonic angle order
    c2, phi = c2[order], phi[order]

    _assert_pointwise_matches_batched(model, c2, phi, mode)
