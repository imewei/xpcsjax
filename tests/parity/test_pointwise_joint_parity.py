"""Phase-0 gate: the flat point-wise heterodyne joint residual must reproduce the
batched joint-fit objective (SSR) to rtol=1e-10 for EVERY per-angle scaling mode,
at a fixed parameter vector. Precondition for routing two_component through the
shared homodyne stratification engine.

This is a DISCOVERY / MEASUREMENT test, not red-green TDD. It measures, at a fixed
``p0``, whether the flat point-wise model SSR equals the batched joint residual SSR
when fed the SAME physics params and the SAME effective per-angle scaling. A failure
on a mode is a valid Phase-0 finding (do not loosen the assertion or touch production
code) — the production code under measurement is:

* ``compute_multi_angle_residuals`` (batched joint residual)
* ``build_heterodyne_pointwise_model`` (flat point-wise model + its scaling map)
* ``FourierReparameterizer`` (fourier scaling reconstruction)
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


def _effective_scaling(mode, p0, meta, n_phi, n_varying, fourier_order=2):
    """(contrast_per_angle, offset_per_angle) in SORTED phi_unique order, matching
    EXACTLY what build_heterodyne_pointwise_model's model_fn computes at p0.

    Reconstructed against the REAL APIs (verified against source):

    * ``fixed_constant`` — frozen quantile scaling lives in ``meta`` (no tail).
    * ``auto_averaged``  — tail = [contrast_scalar, offset_scalar], broadcast.
    * ``individual``     — tail = [contrast(n_phi) | offset(n_phi)].
    * ``fourier``        — tail = full Fourier coefficient vector. The builder
      constructs ``FourierReparameterizer(phi_unique, FourierReparamConfig(
      mode="fourier", fourier_order=fourier_order))`` and the model_fn expands the
      tail via ``fourier_to_per_angle_jax(tail)``. We mirror that here with the
      numpy twin ``fourier_to_per_angle(tail)`` (same basis matrix, returns the
      same ``(contrast, offset)`` tuple), building the reparameterizer IDENTICALLY
      to the builder (same ``phi_unique`` array it closed over).
    """
    tail = np.asarray(p0[n_varying:], dtype=float)
    if mode == "fixed_constant":
        return (
            np.asarray(meta["contrast_arr"], float),
            np.asarray(meta["offset_arr"], float),
        )
    if mode == "auto_averaged":
        return np.full(n_phi, float(tail[0])), np.full(n_phi, float(tail[1]))
    if mode == "individual":
        return tail[:n_phi].copy(), tail[n_phi:].copy()
    if mode == "fourier":
        from xpcsjax.optimization.nlsq.fourier_reparam import (
            FourierReparamConfig,
            FourierReparameterizer,
        )

        # Build the reparameterizer identically to the production builder
        # (heterodyne_hybrid_streaming.build_heterodyne_pointwise_model): same
        # phi_unique array, same config. fourier_to_per_angle consumes the FULL
        # coefficient vector [contrast_coeffs | offset_coeffs] and returns
        # (contrast_per_angle, offset_per_angle) in phi_unique order.
        config = FourierReparamConfig(mode="fourier", fourier_order=fourier_order)
        rep = FourierReparameterizer(np.asarray(meta["phi_unique"], float), config)
        contrast, offset = rep.fourier_to_per_angle(tail)
        return np.asarray(contrast, float), np.asarray(offset, float)
    raise AssertionError(mode)


def _assert_pointwise_matches_batched(model, c2, phi, mode, *, fourier_order=2) -> dict:
    """Run the full point-wise vs batched SSR comparison for one mode at p0.

    Steps (the shared ~7-step plumbing both tests need):
        1. build stratified data
        2. build_heterodyne_pointwise_model
        3. flat point-wise SSR at p0
        4. _effective_scaling -> per-angle (contrast, offset) in phi_unique order
        5. searchsorted reindex from phi_unique order back to caller's phi order
        6. batched compute_multi_angle_residuals SSR at the SAME physics + scaling
        7. assert SSR parity at rtol=1e-10

    Returns ``meta`` so callers can apply extra mode-specific guards inline
    (e.g. the true-Fourier tail-length / fourier_effective_mode checks).
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
        fourier_order=fourier_order,
    )

    pw = np.asarray(model_fn(jnp.asarray(x_data), *p0))
    pw_ssr = float(np.sum((pw - np.asarray(y_data)) ** 2))

    c_sorted, o_sorted = _effective_scaling(
        mode, p0, meta, len(phi), n_varying, fourier_order=fourier_order
    )
    phi_unique = np.asarray(meta["phi_unique"], float)
    pos = np.searchsorted(phi_unique, np.asarray(phi, float))
    contrasts, offsets = c_sorted[pos], o_sorted[pos]

    full = np.asarray(model.param_manager.get_full_values(), dtype=float).copy()
    full[np.asarray(model.param_manager.varying_indices)] = np.asarray(p0[:n_varying])

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


@pytest.mark.parametrize("mode", ["fixed_constant", "auto_averaged", "individual", "fourier"])
def test_pointwise_joint_ssr_matches_batched(mode):
    # NOTE on the "fourier" case here: with n_phi=4 and fourier_order=2,
    # min_angles = 1 + 2*fourier_order = 5 > 4, so FourierReparameterizer sets
    # use_fourier=False and FALLS BACK to independent per-angle scaling
    # (n_coeffs = 2*n_phi = 8, NOT 2*(2*fourier_order+1) = 10). So this
    # parametrized "fourier" case does NOT exercise the true Fourier basis-matrix
    # path — it collapses to the "individual" layout (hence identical SSR to
    # fixed_constant/individual). The dedicated n_phi=6 test below
    # (test_pointwise_joint_ssr_matches_batched_fourier_true_basis) exercises the
    # genuine Fourier basis path, which is the highest-risk mode for the plan.
    model, c2, phi = make_synthetic_two_component(n_phi=4, n_t=16)
    order = np.array([2, 0, 3, 1])  # non-monotonic angle order
    c2, phi = c2[order], phi[order]

    _assert_pointwise_matches_batched(model, c2, phi, mode)


def test_pointwise_joint_ssr_matches_batched_fourier_true_basis():
    """Exercise the GENUINE Fourier basis-matrix path (use_fourier=True).

    The parametrized ``fourier`` case uses n_phi=4, where
    ``min_angles = 1 + 2*fourier_order = 5 > 4`` forces
    ``FourierReparameterizer`` into the independent-mode fallback — so it never
    touches the basis matrix and is numerically identical to ``individual``.
    Here we use n_phi=6 (>= min_angles=5) so ``use_fourier=True`` and the true
    Fourier expansion (B @ coeffs) is what ``model_fn`` evaluates. This is the
    highest-risk mode for routing two_component through the shared homodyne
    stratification engine, so the Phase-0 gate must measure it directly.
    """
    fourier_order = 2
    model, c2, phi = make_synthetic_two_component(n_phi=6, n_t=16)
    order = np.array([2, 5, 0, 4, 3, 1])  # non-monotonic angle order
    c2, phi = c2[order], phi[order]

    meta = _assert_pointwise_matches_batched(model, c2, phi, "fourier", fourier_order=fourier_order)

    # Guard 1 (coefficient count): confirm the TRUE Fourier basis path is active.
    # At n_phi=6 the scaling tail is the Fourier coefficient vector of length
    # 2*(2*fourier_order+1) = 10. The independent-mode fallback would instead give
    # 2*n_phi = 12. Asserting the exact basis-path length makes any future
    # regression back into the fallback (which would silently stop testing the
    # basis matrix) fail loudly.
    expected_fourier_tail = 2 * (2 * fourier_order + 1)  # = 10
    fallback_tail = 2 * len(phi)  # = 12 (independent fallback)
    tail_len = meta["n_scaling"]
    assert tail_len == expected_fourier_tail, (
        f"expected true-Fourier tail length {expected_fourier_tail} "
        f"(use_fourier=True basis path), got {tail_len} "
        f"(== {fallback_tail} would mean the independent-mode fallback ran)"
    )

    # Guard 2 (semantic): pin the basis path via the builder's authoritative
    # field, not just the coefficient count. fourier_effective_mode is "fourier"
    # only when FourierReparameterizer actually used the basis; it reports
    # "individual" on the silent n_phi-too-small fallback.
    assert meta["fourier_effective_mode"] == "fourier", (
        "expected meta['fourier_effective_mode'] == 'fourier' (true basis path), "
        f"got {meta['fourier_effective_mode']!r} (fallback to independent)"
    )
