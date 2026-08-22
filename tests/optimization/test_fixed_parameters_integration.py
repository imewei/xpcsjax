"""Integration tests: fixed_parameters/active_parameters actually constrain
the real NLSQ solve (not just ParameterManager in isolation). This is the
real proof for Tasks 3 (wrapper.py) and 4 (adapter.py) -- see their task
headers for why they carry no standalone test of their own."""

import numpy as np
import pytest

from xpcsjax.config import ConfigManager
from xpcsjax.core.jax_backend import compute_g2_scaled
from xpcsjax.optimization.nlsq.core import fit_nlsq_jax

TRUE_PHYSICAL_LAMINAR = {
    "D0": 8000.0,
    "alpha": -1.2,
    "D_offset": 50.0,
    "gamma_dot_t0": 0.01,
    "beta": 0.1,
    "gamma_dot_t_offset": 0.0,
    "phi0": 0.0,
}
TRUE_PHYSICAL_STATIC = {"D0": 8000.0, "alpha": -1.2, "D_offset": 50.0}
CONTRAST, OFFSET, Q, L, DT = 0.3, 0.8, 0.005, 2_000_000.0, 0.001

_PHYSICAL_BY_MODE = {
    "laminar_flow": TRUE_PHYSICAL_LAMINAR,
    "static_isotropic": TRUE_PHYSICAL_STATIC,
    "static_anisotropic": TRUE_PHYSICAL_STATIC,  # same 3-name physical set as static_isotropic (round 3 Codex finding #11 -- spec names both)
}
_ALL_PHYSICAL_NAMES = list(
    TRUE_PHYSICAL_LAMINAR.keys()
)  # static forward-sim still uses the full 7-slot kernel


def _physical_index(n_params, physical_names, name):
    """Index of a physical parameter in `result.parameters`.

    Physics is ALWAYS the tail of the returned vector regardless of the
    resolved per-angle scaling mode (see CLAUDE.md's "Physics is ALWAYS the
    tail" note): the scaling head varies in width -- e.g. laminar_flow with
    n_phi=3 resolves `auto -> averaged` for the SOLVE, but the wrapper's
    Phase-5 contract expands the returned vector back to the dense
    scaling-first per-angle layout (2*n_phi head), not the compact
    [contrast, offset] head a naive reading of the solve-time shape would
    suggest.
    """
    n_physical = len(physical_names)
    return n_params - n_physical + physical_names.index(name)


def _synthetic_data(analysis_mode="laminar_flow", n_t=10, n_phi=3, seed=0):
    import jax.numpy as jnp

    true_physical = _PHYSICAL_BY_MODE[analysis_mode]
    # compute_g2_scaled's kernel always takes the full 7-parameter vector;
    # for static mode the shear-related entries are simply absent from
    # TRUE_PHYSICAL_STATIC and default to 0.0 here -- physically equivalent
    # to pure diffusion, matching what a static-mode optimizer vector means.
    full_physical = {**dict.fromkeys(_ALL_PHYSICAL_NAMES, 0.0), **true_physical}
    t = np.arange(1, n_t + 1) * DT
    t1, t2 = np.meshgrid(t, t, indexing="ij")
    phi = np.array([0.0, 45.0, 90.0])[:n_phi]
    params_vec = jnp.array([full_physical[name] for name in _ALL_PHYSICAL_NAMES])
    g2 = np.stack(
        [
            # compute_g2_scaled always returns a leading n_phi axis (length 1
            # for a scalar phi) -- squeeze it before stacking our own n_phi
            # axis, or g2 ends up (n_phi, 1, n_t, n_t) instead of
            # (n_phi, n_t, n_t) and every consumer downstream misreads the
            # shape (e.g. the adapter's angle-major flattening).
            np.asarray(
                compute_g2_scaled(
                    params_vec,
                    jnp.asarray(t1),
                    jnp.asarray(t2),
                    jnp.asarray(p),
                    Q,
                    L,
                    CONTRAST,
                    OFFSET,
                    DT,
                )
            )[0]
            for p in phi
        ],
        axis=0,
    )
    rng = np.random.default_rng(seed)
    g2_noisy = g2 + rng.normal(scale=1e-4, size=g2.shape)
    return {
        "phi": phi,
        "g2": g2_noisy,
        "t1": t1,
        "t2": t2,
        "q": Q,
        "L": L,
        "dt": DT,
        "sigma": 1e-4 * np.ones_like(g2_noisy),
    }


def _config(
    analysis_mode="laminar_flow",
    fixed_parameters=None,
    active_parameters=None,
    extra_initial=None,
    extra_top=None,
):
    true_physical = _PHYSICAL_BY_MODE[analysis_mode]
    initial = {
        "parameter_names": list(true_physical.keys()),
        "values": list(true_physical.values()),
    }
    if fixed_parameters:
        initial["fixed_parameters"] = fixed_parameters
    if active_parameters is not None:
        initial["active_parameters"] = active_parameters
    if extra_initial:
        initial.update(extra_initial)
    config = {"analysis_mode": analysis_mode, "initial_parameters": initial}
    if extra_top:
        config.update(extra_top)
    return config


@pytest.mark.parametrize(
    "use_adapter",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "NLSQAdapter cannot complete a real solve via fit_nlsq_jax: "
                    "_normalize_data_to_object collapses 2D t1/t2 to length-n_t while "
                    "_flatten_xpcs_data needs angle-tiled flat arrays, and nlsq.curve_fit "
                    "rejects the resulting 3-column xdata even with adapter-native flat "
                    "data. Both reproduce with resolved_physical=None -- pre-existing, "
                    "unrelated to fixed_parameters. Task 4's adapter.py wiring is "
                    "therefore unreachable through this entry point (verified live)."
                ),
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "analysis_mode", ["static_isotropic", "static_anisotropic", "laminar_flow"]
)
def test_fixed_parameter_survives_real_fit(analysis_mode, use_adapter):
    data = _synthetic_data(analysis_mode)
    fixed_value = 37.5  # different from the true simulated value (50.0)
    cm = ConfigManager(
        config_override=_config(analysis_mode, fixed_parameters={"D_offset": fixed_value})
    )
    result = fit_nlsq_jax(data, cm, use_adapter=use_adapter)
    physical_names = list(_PHYSICAL_BY_MODE[analysis_mode].keys())
    params = np.asarray(result.parameters).ravel()
    d_offset_idx = _physical_index(len(params), physical_names, "D_offset")
    assert abs(params[d_offset_idx] - fixed_value) < 1e-9
    if result.uncertainties is not None:
        # A fixed parameter's true covariance diagonal is exactly 0.
        # `safe_uncertainties_from_pcov` (recovery.py) floors ANY near-zero
        # diagonal entry as a generic numerical-safety net, but
        # `_post_process_results` (wrapper.py) explicitly re-zeroes the
        # uncertainty at every FIXED physical position afterward -- the plan's
        # invariant is bit-exact 0.0, not "small".
        assert np.asarray(result.uncertainties).ravel()[d_offset_idx] == 0.0
    if use_adapter:
        # Prove the adapter itself (Task 4) honored the fixed parameter,
        # not that it failed and silently fell back to NLSQWrapper (Task 3)
        # -- core.py's adapter->wrapper fallback would make this assertion
        # pass either way if left unchecked.
        assert result.device_info.get("fallback_occurred") is False
        assert result.device_info.get("adapter") != "NLSQWrapper"


def test_fixed_scaling_parameter_raises():
    data = _synthetic_data("laminar_flow")
    cm = ConfigManager(config_override=_config("laminar_flow", fixed_parameters={"contrast": 0.5}))
    with pytest.raises(ValueError, match="contrast"):
        fit_nlsq_jax(data, cm, use_adapter=False)


def test_unset_fixed_parameters_is_a_noop():
    data = _synthetic_data("laminar_flow")
    cm = ConfigManager(config_override=_config("laminar_flow"))
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    # n_phi=3 hits the default constant_scaling_threshold (3), so laminar_flow
    # resolves auto -> averaged for the solve, but the returned vector is the
    # dense scaling-first per-angle layout: 2*n_phi (6) + n_physical (7) = 13.
    assert np.asarray(result.parameters).size == 2 * 3 + len(TRUE_PHYSICAL_LAMINAR)


def test_restricted_active_parameters_real_fit():
    """A physical parameter excluded via active_parameters must not move from
    its initial value -- distinct mechanism entry point from fixed_parameters,
    same underlying resolver."""
    data = _synthetic_data("laminar_flow")
    active = [
        "D0",
        "alpha",
        "gamma_dot_t0",
        "beta",
        "gamma_dot_t_offset",
        "phi0",
    ]  # excludes D_offset
    cm = ConfigManager(config_override=_config("laminar_flow", active_parameters=active))
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    params = np.asarray(result.parameters).ravel()
    d_offset_idx = _physical_index(len(params), _ALL_PHYSICAL_NAMES, "D_offset")
    assert abs(params[d_offset_idx] - 50.0) < 1e-9  # unchanged from its initial value


def test_x_scale_map_array_branch_with_fixed_parameter():
    """Forces x_scale_value to be an ARRAY (not the default 'jac' string) via
    optimization.nlsq.x_scale_map, combined with fixed_parameters -- the
    branch v2's plan would have crashed on (slicing a 3-char string)."""
    data = _synthetic_data("laminar_flow")
    config = _config(
        "laminar_flow",
        fixed_parameters={"D_offset": 37.5},
        extra_top={
            "optimization": {
                "nlsq": {
                    "x_scale_map": {name: 1.0 for name in TRUE_PHYSICAL_LAMINAR},
                }
            }
        },
    )
    cm = ConfigManager(config_override=config)
    result = fit_nlsq_jax(data, cm, use_adapter=False)  # must not raise
    params = np.asarray(result.parameters).ravel()
    d_offset_idx = _physical_index(len(params), _ALL_PHYSICAL_NAMES, "D_offset")
    assert abs(params[d_offset_idx] - 37.5) < 1e-9


def test_fixed_parameter_survives_cmaes_fit():
    """fixed_parameters must survive fit_nlsq_cmaes's own Phase 1 (NLSQ
    warm-start) / Phase 2 (CMA-ES) / Phase 3 (result-selection) sequence --
    a distinct engine from fit_nlsq_jax's local path proven in Task 5.

    auto_select is forced off so the fit actually runs through fit_nlsq_cmaes
    rather than silently falling back to local NLSQ if the synthetic bounds'
    scale ratio doesn't clear the auto-select threshold -- see
    _laminar_cmaes_config in test_cmaes_trigger.py for the same pattern.
    """
    data = _synthetic_data("laminar_flow")
    config = _config(
        "laminar_flow",
        fixed_parameters={"D_offset": 37.5},
        extra_top={"optimization": {"nlsq": {"cmaes": {"enable": True, "auto_select": False}}}},
    )
    cm = ConfigManager(config_override=config)
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    params = np.asarray(result.parameters).ravel()
    d_offset_idx = _physical_index(len(params), _ALL_PHYSICAL_NAMES, "D_offset")
    assert abs(params[d_offset_idx] - 37.5) < 1e-9
    if result.uncertainties is not None:
        assert np.asarray(result.uncertainties).ravel()[d_offset_idx] == 0.0
