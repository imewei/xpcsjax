"""Individual per-angle mode is a JOINT fit (parity with laminar_flow + upstream).

Heterodyne ``two_component`` ``individual`` per-angle mode previously ran
*sequential* per-angle fits and reported ``mean(physics)`` as ``parameters``
while reporting ``chi_squared`` as the sum of each angle's *own*-physics SSR.
That is an internally inconsistent estimator: the returned parameter vector
does NOT reproduce the reported ``chi_squared``.

The fix routes explicit multi-angle ``individual`` through the existing joint
solver (``_fit_joint_multi_phi`` via ``FourierReparameterizer`` independent
mode), exactly like the ``fourier`` branch and matching xpcsjax
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
    ``(contrast, offset)`` read from the individual-mode scaling tail
    ``[physics | contrast_0..contrast_{n_phi-1} | offset_0..offset_{n_phi-1}]``.
    Sums the off-diagonal / t=0-excluded residual SSR across all angles —
    the same masked support ``compute_residuals`` (and therefore the joint
    fit) uses.
    """
    pm = model.param_manager
    n_physics = int(pm.n_varying)
    n_phi = len(phi)
    params = np.asarray(params, dtype=np.float64)

    physics_varying = params[:n_physics]
    contrasts = params[n_physics : n_physics + n_phi]
    offsets = params[n_physics + n_phi : n_physics + 2 * n_phi]
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

    # Parameter layout is [physics | 2 * n_phi per-angle scaling].
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
