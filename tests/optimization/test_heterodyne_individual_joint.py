"""Individual per-angle mode is a JOINT fit (parity with laminar_flow + upstream).

Heterodyne ``two_component`` ``individual`` per-angle mode previously ran
*sequential* per-angle fits and reported ``mean(physics)`` as ``parameters``
while reporting ``chi_squared`` as the sum of each angle's *own*-physics SSR.
That is an internally inconsistent estimator: the returned parameter vector
does NOT reproduce the reported ``chi_squared``.

The fix routes explicit multi-angle ``individual`` through the existing joint
solver (``_fit_joint_multi_phi`` with per-angle scaling layout),
matching xpcsjax
``laminar_flow`` and upstream heterodyne. A correct joint fit has one
consistent optimum, so re-evaluating the model at ``res.parameters`` MUST
reproduce ``res.chi_squared``.

The decisive test below re-evaluates the joint data residual at the returned
parameters and asserts the resulting SSR equals the reported ``chi_squared``.
Against the old sequential aggregate this FAILS (mean-physics params do not
reproduce the per-angle-own-physics chi-sum); against the joint path it PASSES.
"""

from __future__ import annotations

import numpy as np

from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.core.heterodyne_jax_backend import compute_residuals
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi


def _reevaluate_joint_ssr(
    model,
    c2: np.ndarray,
    phi: np.ndarray,
    params: np.ndarray,
) -> float:
    """Re-evaluate the joint data-residual SSR at ``params``.

    Independently reconstructs the residual the joint solver minimizes:
    physics block expanded to the full parameter vector, with per-angle
    ``(contrast, offset)`` read from the canonical SCALING-FIRST individual-mode
    layout ``[contrast_0..contrast_{n_phi-1} | offset_0..offset_{n_phi-1} |
    physics]`` (scaling head, physics tail). Sums the off-diagonal /
    t=0-excluded residual SSR across all angles — the same masked support
    ``compute_residuals`` (and therefore the joint fit) uses.
    """
    pm = model.param_manager
    n_phi = len(phi)
    params = np.asarray(params, dtype=np.float64)

    contrasts = params[:n_phi]
    offsets = params[n_phi : 2 * n_phi]
    physics_varying = params[2 * n_phi :]
    full_physics = np.asarray(pm.expand_varying_to_full(physics_varying), dtype=np.float64)

    total = 0.0
    for i in range(n_phi):
        r = np.asarray(
            compute_residuals(
                full_physics,
                model.t,
                model.q,
                model.dt,
                float(phi[i]),
                c2[i],
                None,
                float(contrasts[i]),
                float(offsets[i]),
            )
        )
        total += float(np.sum(r**2))
    return total


def test_individual_mode_is_joint_params_reproduce_chi2():
    """A joint individual fit's parameters MUST reproduce its reported chi_squared."""
    model, c2, phi = make_synthetic_two_component(n_phi=4, n_t=20)
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "individual"})
    res = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)
    diag = res.nlsq_diagnostics
    assert diag is not None

    assert diag["per_angle_mode"] == "individual"

    # JOINT, not the sequential aggregate.
    assert diag.get("covariance_structure") != "block_diagonal_sequential", (
        "individual multi-angle must be a joint fit, not the sequential aggregate"
    )

    # Parameter layout is canonical scaling-first [2 * n_phi per-angle scaling | physics].
    n_physics = int(model.param_manager.n_varying)
    assert res.parameters.shape == (n_physics + 2 * len(phi),)

    # SSR conservation: chi2_per_angle.sum() == chi_squared (joint invariant).
    np.testing.assert_allclose(np.asarray(diag["chi2_per_angle"]).sum(), res.chi_squared, rtol=1e-6)

    # CONSISTENCY: re-evaluating the joint residual at res.parameters
    # reproduces chi_squared. The sequential aggregate (mean physics) breaks
    # this; a correct joint optimum satisfies it.
    reps_ssr = _reevaluate_joint_ssr(model, c2, phi, np.asarray(res.parameters))
    np.testing.assert_allclose(reps_ssr, res.chi_squared, rtol=1e-6)


def test_individual_single_angle_still_falls_back_to_sequential():
    """Single-angle individual stays on the sequential aggregate (legit fallback)."""
    model, c2, phi = make_synthetic_two_component(n_phi=1, n_t=20)
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "individual"})
    res = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)
    diag = res.nlsq_diagnostics
    assert diag is not None
    assert diag["per_angle_mode"] == "individual"
    # n_phi <= 1 → sequential aggregate is the legitimate fallback.
    assert diag.get("covariance_structure") == "block_diagonal_sequential"


def test_individual_no_config_falls_back_to_sequential():
    """config is None → sequential per-angle aggregate (legit fallback)."""
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    res = fit_nlsq_multi_phi(model, c2, phi, config=None, weights=None)
    diag = res.nlsq_diagnostics
    assert diag is not None
    assert diag.get("covariance_structure") == "block_diagonal_sequential"

    # Debug-audit fix: _aggregate_individual_results must report the
    # noise-normalized reduced chi2 (matching every sibling path), not raw
    # SSR/dof (which collapses to a statistically meaningless MSE << 1 on
    # normalized C2 data). Recompute independently via the same shared
    # helper the fix wires in, using the code's own n_data_total formula
    # ((N-1)*(N-2) per angle) and total_dim = len(res.parameters).
    from xpcsjax.optimization.nlsq.heterodyne_data_prep import (
        noise_normalized_reduced_chi2,
    )

    n_t = c2.shape[1]
    n_data_total = int(len(phi)) * max(n_t - 1, 0) * max(n_t - 2, 0)
    total_dim = len(res.parameters)
    expected = noise_normalized_reduced_chi2(
        ssr=float(res.chi_squared),
        c2_data=np.asarray(c2),
        n_data_valid=n_data_total,
        n_params=total_dim,
    )
    assert np.isclose(res.reduced_chi_squared, expected, rtol=1e-9)
    # And it must actually differ from the pre-fix raw-MSE value (unless the
    # far-lag noise variance happens to be ~1.0, which this fixture's noise
    # level does not produce).
    raw_mse = float(res.chi_squared) / max(n_data_total - total_dim, 1)
    assert not np.isclose(res.reduced_chi_squared, raw_mse, rtol=1e-3)
