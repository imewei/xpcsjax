"""Tests for ParameterManager.expand_reduced_result."""

from __future__ import annotations

import numpy as np

from xpcsjax.config.heterodyne_parameter_manager import ParameterManager
from xpcsjax.config.heterodyne_parameter_names import ALL_PARAM_NAMES
from xpcsjax.config.heterodyne_parameter_space import ParameterSpace


def _manager(tied=None, active=None) -> ParameterManager:
    initial_parameters = {}
    if tied:
        initial_parameters["tied_parameters"] = tied
    config = {"analysis_mode": "two_component"}
    if initial_parameters:
        config["initial_parameters"] = initial_parameters
    space = ParameterSpace.from_config(config)
    pm = ParameterManager(space)
    if active is not None:
        # Set parameters not in active list as non-varying
        for param_name in ALL_PARAM_NAMES:
            if param_name not in active:
                pm.set_vary(param_name, False)
    return pm


def test_expand_reduced_result_untied_physics_first_no_scaling():
    """Fully-untied 'constant' mode: n_scaling=0, physics_first irrelevant."""
    pm = _manager()
    n_varying = len(pm.varying_indices)
    params_reduced = np.arange(n_varying, dtype=np.float64)
    cov_reduced = np.eye(n_varying, dtype=np.float64)
    unc_reduced = np.ones(n_varying, dtype=np.float64)

    params_full, cov_full, unc_full = pm.expand_reduced_result(
        params_reduced, cov_reduced, unc_reduced, n_scaling=0, scaling_first=False
    )
    assert params_full.shape == (14,)
    assert cov_full.shape == (14, 14)
    assert unc_full.shape == (14,)
    assert np.all(np.isfinite(params_full))
    # every varying physics param round-trips its own value
    for a, idx in enumerate(pm.varying_indices):
        assert params_full[idx] == params_reduced[a]
        assert unc_full[idx] == unc_reduced[a]
        assert cov_full[idx, idx] == cov_reduced[a, a]


def test_expand_reduced_result_physics_first_with_scaling():
    pm = _manager()
    n_varying = len(pm.varying_indices)
    n_scaling = 2
    n = n_varying + n_scaling
    params_reduced = np.arange(n, dtype=np.float64)
    cov_reduced = np.eye(n, dtype=np.float64) * 2.0
    unc_reduced = np.full(n, 3.0)

    params_full, cov_full, unc_full = pm.expand_reduced_result(
        params_reduced, cov_reduced, unc_reduced, n_scaling=n_scaling, scaling_first=False
    )
    assert params_full.shape == (14 + n_scaling,)
    # scaling tail (last 2 entries) pass through unchanged
    assert np.array_equal(params_full[-n_scaling:], params_reduced[-n_scaling:])
    assert unc_full[-1] == 3.0
    assert cov_full[-1, -1] == 2.0


def test_expand_reduced_result_scaling_first_with_scaling():
    pm = _manager()
    n_varying = len(pm.varying_indices)
    n_scaling = 4
    n = n_varying + n_scaling
    params_reduced = np.arange(n, dtype=np.float64)

    params_full, cov_full, unc_full = pm.expand_reduced_result(
        params_reduced, None, None, n_scaling=n_scaling, scaling_first=True
    )
    assert params_full.shape == (14 + n_scaling,)
    # scaling head (first n_scaling entries) pass through unchanged
    assert np.array_equal(params_full[:n_scaling], params_reduced[:n_scaling])
    assert cov_full.shape == (14 + n_scaling, 14 + n_scaling)
    assert np.all(np.isnan(cov_full))  # None input -> all-NaN output
    assert np.all(np.isnan(unc_full))


def test_expand_reduced_result_fixed_physics_param_gets_nan():
    """A physics param excluded via active_parameters (not tied) must be NaN
    in the expanded covariance/uncertainty, not crash and not silently 0."""
    pm = _manager(active=[n for n in ALL_PARAM_NAMES if n != "D0_ref"])
    n_varying = len(pm.varying_indices)
    params_reduced = np.ones(n_varying, dtype=np.float64) * 7.0
    cov_reduced = np.eye(n_varying, dtype=np.float64)
    unc_reduced = np.ones(n_varying, dtype=np.float64)

    params_full, cov_full, unc_full = pm.expand_reduced_result(
        params_reduced, cov_reduced, unc_reduced, n_scaling=0, scaling_first=False
    )
    d0_ref_idx = list(ALL_PARAM_NAMES).index("D0_ref")
    assert np.isnan(unc_full[d0_ref_idx])
    assert np.isnan(cov_full[d0_ref_idx, d0_ref_idx])
    # the fixed value itself IS reported (from expand_varying_to_full), just
    # with no computed uncertainty
    assert np.isfinite(params_full[d0_ref_idx])


def test_expand_reduced_result_tied_child_mirrors_parent_covariance():
    pm = _manager(tied={"D0_ref": "D0_sample"})
    n_varying = len(pm.varying_indices)
    n_scaling = 2
    n = n_varying + n_scaling
    rng = np.random.default_rng(0)
    params_reduced = rng.normal(size=n)
    a = rng.normal(size=(n, n))
    cov_reduced = a @ a.T  # symmetric positive-semidefinite
    unc_reduced = np.sqrt(np.clip(np.diag(cov_reduced), 0, None))

    params_full, cov_full, unc_full = pm.expand_reduced_result(
        params_reduced, cov_reduced, unc_reduced, n_scaling=n_scaling, scaling_first=False
    )
    d0_ref_idx = list(ALL_PARAM_NAMES).index("D0_ref")
    d0_sample_idx = list(ALL_PARAM_NAMES).index("D0_sample")

    assert params_full[d0_ref_idx] == params_full[d0_sample_idx]
    assert unc_full[d0_ref_idx] == unc_full[d0_sample_idx]
    assert cov_full[d0_ref_idx, d0_ref_idx] == cov_full[d0_sample_idx, d0_sample_idx]
    # child's covariance with a THIRD parameter (v0) equals parent's
    v0_idx = list(ALL_PARAM_NAMES).index("v0")
    assert cov_full[d0_ref_idx, v0_idx] == cov_full[d0_sample_idx, v0_idx]
    # child's covariance with the scaling tail equals parent's
    assert cov_full[d0_ref_idx, -1] == cov_full[d0_sample_idx, -1]
