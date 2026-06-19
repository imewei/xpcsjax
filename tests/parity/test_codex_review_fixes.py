"""Regression tests for the two confirmed Codex adversarial-review findings.

Both are Phase-5 compressed-layout regressions: the averaged/constant optimizer
vector is scaling-first compressed (``[scaling_head | physics]`` with
``len(scaling_head) == n_optimized`` = 0 for constant, 2 for averaged), but two
helper paths were left keyed on the DENSE ``2*n_phi`` head.

Finding 1 (transform index map): ``build_physical_index_map`` placed physics at
``2*n_phi + idx``; applied to a compressed averaged/constant vector this indexes
out of bounds when shear transforms are enabled
(``optimization.nlsq.shear_transforms.enable_gamma_dot_log``).

Finding 2 (large-path DOF): the out-of-core / hybrid-streaming / stratified-LS
reduced-chi2 ladders only expanded DOF for ``auto`` (and ``constant``), so an
EXPLICIT ``per_angle_mode: averaged`` fell through to ``len(popt)`` instead of the
constrained-model ``2*n_phi + n_physical``.

The x_scale sub-claim of Finding 1 was a FALSE POSITIVE (x_scale is resized to the
compressed length before the solver) and is not tested here.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.per_angle_mode import (
    effective_constrained_dof,
    resolve_per_angle_mode,
)
from xpcsjax.optimization.nlsq.transforms import (
    apply_forward_shear_transforms_to_vector,
    build_physical_index_map,
)

_PHYS = ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]


# ---------------------------------------------------------------------------
# Finding 1 — transform index map must follow the compressed scaling head.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "head_size,vec_len",
    [
        (0, len(_PHYS)),  # constant: [*physics]
        (2, 2 + len(_PHYS)),  # averaged: [c_avg, o_avg, *physics]
        (8, 8 + len(_PHYS)),  # individual n_phi=4: [c0..c3, o0..o3, *physics]
    ],
)
def test_physical_index_map_honours_scaling_head_size(head_size, vec_len):
    """With ``scaling_head_size`` given, physics indices start at the head end and
    stay in-bounds for the compressed vector, for every per-angle mode."""
    idx_map = build_physical_index_map(True, 4, _PHYS, scaling_head_size=head_size)
    assert idx_map["D0"] == head_size
    assert idx_map["phi0"] == head_size + len(_PHYS) - 1
    # every physics index must be addressable in the compressed vector
    assert max(idx_map.values()) < vec_len


def test_physical_index_map_default_is_dense_backward_compatible():
    """Without ``scaling_head_size`` the legacy dense ``2*n_phi`` head is preserved
    (this is what the static/individual golden path relies on -> rtol=1e-10)."""
    idx_map = build_physical_index_map(True, 4, _PHYS)
    assert idx_map["D0"] == 8  # 2*n_phi
    idx_map_compact = build_physical_index_map(False, 4, _PHYS)
    assert idx_map_compact["D0"] == 2


@pytest.mark.parametrize("_mode,head_size", [("constant", 0), ("averaged", 2)])
def test_forward_shear_transform_on_compressed_vector_no_indexerror(_mode, head_size):
    """The exact crash from the review: a forward shear transform on a compressed
    averaged/constant vector must NOT raise IndexError once the index map follows
    the compressed head."""
    n_phys = len(_PHYS)
    vec = np.concatenate([np.full(head_size, 0.5), np.linspace(1.0, 7.0, n_phys)])
    idx_map = build_physical_index_map(True, 4, _PHYS, scaling_head_size=head_size)
    cfg = {"enable_gamma_dot_log": True, "enable_beta_centering": True, "beta_reference": 0.1}
    out, _state = apply_forward_shear_transforms_to_vector(vec, idx_map, cfg)
    assert out.shape == vec.shape
    assert np.all(np.isfinite(out))
    # gamma_dot_t0 (4th physical) is log-transformed in place at its compressed index
    gi = idx_map["gamma_dot_t0"]
    assert np.isclose(out[gi], np.log(vec[gi]))


def _laminar_fit(mode, *, shear: bool, n_phi: int = 4, n_t: int = 10):
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.homodyne_model import HomodyneModel
    from xpcsjax.optimization.nlsq import fit_nlsq

    phi = np.linspace(0.0, 90.0, n_phi)
    t = np.linspace(0.0, float(n_t - 1), n_t)
    true = np.array([1000.0, 0.5, 10.0, 0.01, 0.5, 0.01, 0.0])
    nlsq = {
        "analysis_mode": "laminar_flow",
        "max_iterations": 30,
        "loss": "linear",
        "cmaes": {"enable": False, "auto_select": False},
        "multi_start": {"enable": False},
        "anti_degeneracy": {
            "enable": True,
            "per_angle_mode": mode,
            "constant_scaling_threshold": 3,
        },
    }
    if shear:
        nlsq["shear_transforms"] = {"enable_gamma_dot_log": True}
    cfg = ConfigManager(
        config_override={
            "analysis_mode": "laminar_flow",
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": n_t,
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": n_t},
                "scattering": {"wavevector_q": 0.0237},
                "geometry": {"stator_rotor_gap": 2000000},
            },
            "initial_parameters": {"parameter_names": list(_PHYS), "values": true.tolist()},
            "optimization": {
                "method": "nlsq",
                "nlsq": nlsq,
                "stratification": {"enabled": False},
            },
        }
    )
    model = HomodyneModel(cfg.config)
    c2 = np.asarray(model.compute_c2(true, phi, contrast=0.3, offset=1.0))
    data = {
        "phi_angles_list": phi,
        "c2_exp": c2,
        "t1": t,
        "t2": t,
        "wavevector_q_list": np.array([0.0237]),
    }
    return fit_nlsq(data, cfg)


@pytest.mark.parametrize("mode", ["individual", "auto", "constant"])
def test_end_to_end_shear_transform_completes_finite(mode):
    """End-to-end guard for BOTH transform fixes: a laminar fit with config-enabled
    shear transforms must complete with FINITE results for every per-angle mode.

    Pre-fix this hit two bugs in sequence:
    1. (Finding 1, compressed modes) dense index map -> ``IndexError`` index 11 vs
       size 9 for averaged/constant.
    2. (JIT-safety, all modes) the transform-wrapped residual called the numpy
       inverse on a JAX tracer -> ``TracerArrayConversionError`` -> non-finite for
       individual/averaged/constant alike.

    With the mode-aware index map (Fix 1b) and the jnp inverse
    (:func:`apply_inverse_shear_transforms_to_vector_jax`) the full fit converges.
    """
    res = _laminar_fit(mode, shear=True)
    assert np.all(np.isfinite(np.asarray(res.parameters))), f"{mode}: non-finite params"
    assert np.isfinite(float(res.chi_squared)), f"{mode}: non-finite chi_squared"


def test_jax_inverse_transform_matches_numpy():
    """The JIT-safe jnp inverse must be numerically identical to the numpy inverse
    (same exp / add math, functional ``.at[].set`` instead of in-place)."""
    from xpcsjax.optimization.nlsq.transforms import (
        apply_inverse_shear_transforms_to_vector,
        apply_inverse_shear_transforms_to_vector_jax,
    )

    solver_vec = np.array([0.3, 0.8, 1000.0, 0.5, np.log(524.0), 2.5, 0.0, 1.0, 0.0])
    state = {"gamma_log_idx": 4, "beta_center_idx": 5, "beta_reference": 0.1}
    np_out = apply_inverse_shear_transforms_to_vector(solver_vec, state)
    jax_out = np.asarray(apply_inverse_shear_transforms_to_vector_jax(solver_vec, state))
    np.testing.assert_allclose(jax_out, np_out, rtol=1e-12, atol=0.0)
    # empty/None state is a pass-through for both
    assert np.array_equal(
        np.asarray(apply_inverse_shear_transforms_to_vector_jax(solver_vec, None)),
        solver_vec,
    )


# ---------------------------------------------------------------------------
# Finding 2 — large-path reduced-chi2 DOF must honour EXPLICIT averaged.
# ---------------------------------------------------------------------------


def test_effective_constrained_dof_rule():
    """The single DOF authority the out-of-core / hybrid-streaming / stratified-LS
    ladders share: averaged -> 2*n_phi + n_physical (expanded constrained DOF),
    constant -> n_physical, individual -> None (caller uses len(popt))."""
    assert effective_constrained_dof("averaged", n_phi=4, n_physical=7) == 2 * 4 + 7
    assert effective_constrained_dof("constant", n_phi=4, n_physical=7) == 7
    assert effective_constrained_dof("individual", n_phi=4, n_physical=7) is None


def test_explicit_averaged_resolves_like_auto_for_dof():
    """The Finding-2 root cause: the broken ladders keyed on ``== 'auto'`` and missed
    EXPLICIT averaged. Resolving first makes explicit ``averaged`` and ``auto`` (at
    n_phi>=threshold) yield the SAME expanded DOF."""
    n_phi, n_phys, thr = 4, 7, 3
    dof_auto = effective_constrained_dof(
        resolve_per_angle_mode("auto", n_phi, thr), n_phi=n_phi, n_physical=n_phys
    )
    dof_explicit = effective_constrained_dof(
        resolve_per_angle_mode("averaged", n_phi, thr), n_phi=n_phi, n_physical=n_phys
    )
    assert dof_auto == dof_explicit == 2 * n_phi + n_phys


# ---------------------------------------------------------------------------
# Round 2 (second adversarial review) — heterodyne large-path reduced-chi2 DOF.
# ---------------------------------------------------------------------------


def test_heterodyne_streaming_averaged_reduced_chi2_uses_expanded_dof():
    """Codex round-2 F1: heterodyne averaged streaming popt is COMPRESSED
    ``[c_avg, o_avg, physics]`` (n_physics + 2), but build_hybrid_streaming_result
    must compute reduced chi^2 with the EXPANDED constrained-model DOF
    ``2*n_phi + n_physics`` (spec §5 decision 3), not ``len(popt)``. Pre-fix the
    builder used ``n_data - len(popt)`` -> n_dof too large -> reduced chi^2 too
    optimistic for averaged."""
    import sys

    sys.path.insert(0, "tests/optimization")
    from test_heterodyne_hybrid_streaming import _make_synthetic_heterodyne

    from xpcsjax.optimization.nlsq.heterodyne_result_builder import (
        build_hybrid_streaming_result,
    )
    from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
        build_heterodyne_stratified_data,
    )
    from xpcsjax.optimization.nlsq.strategies.heterodyne_hybrid_streaming import (
        fit_with_stratified_hybrid_streaming_heterodyne,
    )

    n_phi, n_t = 4, 8
    model, c2, phi = _make_synthetic_heterodyne(n_phi=n_phi, n_t=n_t)
    strat = build_heterodyne_stratified_data(model, c2, phi, weights=None)
    lo, hi = model.param_manager.get_bounds()
    popt, pcov, info = fit_with_stratified_hybrid_streaming_heterodyne(
        stratified_data=strat,
        model=model,
        physical_param_names=list(model.param_manager.varying_names),
        initial_params=np.asarray(model.param_manager.get_initial_values(), dtype=np.float64),
        bounds=(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)),
        hybrid_config={"verbose": 0},
        anti_degeneracy_config={"per_angle_mode": "auto"},  # -> averaged at n_phi=4
    )
    assert info["anti_degeneracy"]["per_angle_mode"] == "averaged"
    # averaged popt is compressed (2 + n_physics), NOT dense (2*n_phi + n_physics)
    n_physics = len(model.param_manager.varying_names)
    assert len(popt) == 2 + n_physics

    # Override SSR/noise/n_data with controlled, DOF-sensitive values (the synthetic
    # fit converges to ~0 residual, where any DOF gives reduced_chi2 ~ 0).
    info = dict(info)
    info["cost"] = 50.0  # ssr = 2*cost = 100
    info["sigma2_noise"] = 1.0
    info["n_data_points"] = 1000
    result = build_hybrid_streaming_result(
        model=model, popt=popt, pcov=pcov, info=info, phi_angles=phi
    )
    ssr = 100.0
    expanded_dof = 1000 - (2 * n_phi + n_physics)  # constrained-model DOF (correct)
    compressed_dof = 1000 - len(popt)  # the pre-fix (wrong) DOF
    expected = ssr / expanded_dof
    wrong = ssr / compressed_dof
    assert not np.isclose(expected, wrong, rtol=1e-6)  # fixture distinguishes the two
    # The builder must use the EXPANDED constrained DOF, NOT len(popt) = 2 + n_physics.
    assert np.isclose(float(result.reduced_chi_squared), expected, rtol=1e-9, atol=0.0), (
        f"reduced_chi2 {result.reduced_chi_squared!r} uses compressed DOF "
        f"({wrong!r}) instead of expanded ({expected!r})"
    )
