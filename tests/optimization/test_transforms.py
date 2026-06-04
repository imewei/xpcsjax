"""Scientific tests for xpcsjax.optimization.nlsq.transforms.

The shear transforms are bijective maps between physical and solver space:
    gamma_dot_t0 -> log(gamma_dot_t0)   (inverse: exp)
    beta         -> beta - beta_ref     (inverse: + beta_ref)

The central correctness property is **round-trip identity**
(``inverse(forward(x)) == x``), which we assert at rtol=1e-12 rather than
hard-coding transformed values. Covariance propagation is checked against the
analytic Jacobian (``dx/dy = exp(y) = x`` for the log transform).
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.optimization.nlsq import transforms as tr

# ---------------------------------------------------------------------------
# Name / scale-map normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("gamma_dot_0", "gamma_dot_t0"),
        ("gamma_dot_t_0", "gamma_dot_t0"),
        ("gamma_dot_offset", "gamma_dot_t_offset"),
        ("phi_0", "phi0"),
        ("D0", "D0"),  # non-aliased passes through
        ("  beta  ", "beta"),  # whitespace stripped
        (None, ""),
        ("", ""),
    ],
)
def test_normalize_param_key(raw: str | None, expected: str) -> None:
    assert tr.normalize_param_key(raw) == expected


def test_normalize_x_scale_map_applies_aliases_and_skips_bad() -> None:
    out = tr.normalize_x_scale_map(
        {"gamma_dot_0": 2.0, "phi_0": "3.5", "bad": "not-a-number", "": 9.0}
    )
    assert out == {"gamma_dot_t0": 2.0, "phi0": 3.5}


def test_normalize_x_scale_map_non_dict() -> None:
    assert tr.normalize_x_scale_map([1, 2, 3]) == {}
    assert tr.normalize_x_scale_map(None) == {}


# ---------------------------------------------------------------------------
# build_per_parameter_x_scale
# ---------------------------------------------------------------------------


def test_x_scale_all_unity_returns_none() -> None:
    out = tr.build_per_parameter_x_scale(
        per_angle_scaling=False,
        n_angles=2,
        physical_param_names=["D0", "alpha", "D_offset"],
        analysis_mode="static_isotropic",
        override_map={},
    )
    assert out is None


def test_x_scale_laminar_defaults_applied() -> None:
    names = ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset"]
    out = tr.build_per_parameter_x_scale(
        per_angle_scaling=False,
        n_angles=1,
        physical_param_names=names,
        analysis_mode="laminar_flow",
        override_map={},
    )
    assert out is not None
    # Layout: [contrast, offset, <physical...>]
    physical = out[2:]
    assert physical[names.index("gamma_dot_t0")] == pytest.approx(524.0)
    assert physical[names.index("beta")] == pytest.approx(4.0)
    assert physical[names.index("gamma_dot_t_offset")] == pytest.approx(771.0)


def test_x_scale_per_angle_layout() -> None:
    out = tr.build_per_parameter_x_scale(
        per_angle_scaling=True,
        n_angles=3,
        physical_param_names=["D0"],
        analysis_mode="static_isotropic",
        override_map={"contrast": 2.0, "offset": 0.5, "D0": 10.0},
    )
    assert out is not None
    # 3 contrast + 3 offset + 1 physical
    assert out.shape == (7,)
    assert np.allclose(out[:3], 2.0)
    assert np.allclose(out[3:6], 0.5)
    assert out[6] == pytest.approx(10.0)


def test_x_scale_per_angle_zero_angles_returns_none() -> None:
    out = tr.build_per_parameter_x_scale(
        per_angle_scaling=True,
        n_angles=0,
        physical_param_names=["D0"],
        analysis_mode="static_isotropic",
        override_map={"D0": 5.0},  # non-unity so we reach the n_angles<=0 guard
    )
    assert out is None


def test_format_x_scale_for_log() -> None:
    assert tr.format_x_scale_for_log(np.zeros(5)) == "array(len=5)"
    assert tr.format_x_scale_for_log(3.0) == "3.0"


# ---------------------------------------------------------------------------
# parse_shear_transform_config / build_physical_index_map
# ---------------------------------------------------------------------------


def test_parse_shear_transform_config_defaults() -> None:
    assert tr.parse_shear_transform_config(None) == {
        "enable_gamma_dot_log": False,
        "enable_beta_centering": False,
        "beta_reference": 0.0,
    }


def test_parse_shear_transform_config_values() -> None:
    out = tr.parse_shear_transform_config(
        {"enable_gamma_dot_log": 1, "enable_beta_centering": True, "beta_reference": 2.5}
    )
    assert out == {
        "enable_gamma_dot_log": True,
        "enable_beta_centering": True,
        "beta_reference": 2.5,
    }


@pytest.mark.parametrize("per_angle", [True, False])
def test_build_physical_index_map(per_angle: bool) -> None:
    names = ["D0", "alpha", "gamma_dot_t0", "beta"]
    out = tr.build_physical_index_map(per_angle, n_angles=3, physical_param_names=names)
    start = 6 if per_angle else 2
    assert out == {n: start + i for i, n in enumerate(names)}


# ---------------------------------------------------------------------------
# Forward / inverse round-trip (the core scientific invariant)
# ---------------------------------------------------------------------------


def _laminar_index_map() -> dict[str, int]:
    names = ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset"]
    return tr.build_physical_index_map(False, 1, names)


def test_forward_inverse_roundtrip_identity() -> None:
    idx = _laminar_index_map()
    cfg = {
        "enable_gamma_dot_log": True,
        "enable_beta_centering": True,
        "beta_reference": 1.5,
    }
    # [contrast, offset, D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset]
    params = np.array([0.3, 1.0, 1e-3, 0.9, 1e-4, 100.0, 2.0, 5.0])
    transformed, state = tr.apply_forward_shear_transforms_to_vector(params, idx, cfg)

    # Forward actually changed the gamma + beta slots.
    assert transformed[idx["gamma_dot_t0"]] == pytest.approx(np.log(100.0))
    assert transformed[idx["beta"]] == pytest.approx(2.0 - 1.5)

    recovered = tr.apply_inverse_shear_transforms_to_vector(transformed, state)
    np.testing.assert_allclose(recovered, params, rtol=1e-12, atol=0.0)


def test_forward_no_transforms_returns_empty_state() -> None:
    idx = _laminar_index_map()
    params = np.array([0.3, 1.0, 1e-3, 0.9, 1e-4, 100.0, 2.0, 5.0])
    out, state = tr.apply_forward_shear_transforms_to_vector(
        params, idx, {"enable_gamma_dot_log": False, "enable_beta_centering": False}
    )
    assert state == {}
    np.testing.assert_array_equal(out, params)


def test_forward_rejects_nonpositive_gamma() -> None:
    idx = _laminar_index_map()
    params = np.array([0.3, 1.0, 1e-3, 0.9, 1e-4, 0.0, 2.0, 5.0])
    with pytest.raises(ValueError, match="gamma_dot_t0 must be > 0"):
        tr.apply_forward_shear_transforms_to_vector(params, idx, {"enable_gamma_dot_log": True})


def test_inverse_with_empty_state_is_identity() -> None:
    params = np.array([1.0, 2.0, 3.0])
    out = tr.apply_inverse_shear_transforms_to_vector(params, None)
    np.testing.assert_array_equal(out, params)
    out2 = tr.apply_inverse_shear_transforms_to_vector(params, {})
    np.testing.assert_array_equal(out2, params)


# ---------------------------------------------------------------------------
# Bounds transforms
# ---------------------------------------------------------------------------


def test_forward_bounds_roundtrip() -> None:
    idx = _laminar_index_map()
    cfg = {
        "enable_gamma_dot_log": True,
        "enable_beta_centering": True,
        "beta_reference": 1.0,
    }
    params = np.array([0.3, 1.0, 1e-3, 0.9, 1e-4, 50.0, 2.0, 5.0])
    _, state = tr.apply_forward_shear_transforms_to_vector(params, idx, cfg)

    lower = np.full(8, 0.1)
    upper = np.full(8, 200.0)
    transformed_bounds = tr.apply_forward_shear_transforms_to_bounds((lower, upper), state)
    assert transformed_bounds is not None
    tl, tu = transformed_bounds
    g = idx["gamma_dot_t0"]
    assert tl[g] == pytest.approx(np.log(0.1))
    assert tu[g] == pytest.approx(np.log(200.0))
    b = idx["beta"]
    assert tl[b] == pytest.approx(0.1 - 1.0)
    assert tu[b] == pytest.approx(200.0 - 1.0)


def test_forward_bounds_none_or_empty_state_passthrough() -> None:
    assert tr.apply_forward_shear_transforms_to_bounds(None, {"x": 1}) is None
    bounds = (np.zeros(3), np.ones(3))
    assert tr.apply_forward_shear_transforms_to_bounds(bounds, {}) is bounds


def test_forward_bounds_rejects_nonpositive_gamma() -> None:
    state = {"gamma_log_idx": 0, "beta_center_idx": None, "beta_reference": 0.0}
    with pytest.raises(ValueError, match="bounds must be > 0"):
        tr.apply_forward_shear_transforms_to_bounds((np.array([-1.0]), np.array([1.0])), state)


# ---------------------------------------------------------------------------
# Covariance adjustment (Jacobian propagation)
# ---------------------------------------------------------------------------


def test_adjust_covariance_log_jacobian() -> None:
    # For y = log(x), dx/dy = x. Covariance scales by the diagonal Jacobian:
    # C_x[i,j] = J_i C_y[i,j] J_j, with J_gamma = physical gamma value.
    cov = np.array([[4.0, 1.0], [1.0, 9.0]])
    physical = np.array([3.0, 7.0])
    transformed = np.array([np.log(3.0), 7.0])
    state = {"gamma_log_idx": 0, "beta_center_idx": None}
    adj = tr.adjust_covariance_for_transforms(cov, transformed, physical, state)
    assert adj[0, 0] == pytest.approx(4.0 * 3.0 * 3.0)
    assert adj[0, 1] == pytest.approx(1.0 * 3.0)
    assert adj[1, 0] == pytest.approx(1.0 * 3.0)
    assert adj[1, 1] == pytest.approx(9.0)  # untouched


def test_adjust_covariance_beta_unchanged() -> None:
    # beta centering derivative is 1, so covariance is unchanged.
    cov = np.array([[4.0, 1.0], [1.0, 9.0]])
    state = {"gamma_log_idx": None, "beta_center_idx": 1}
    adj = tr.adjust_covariance_for_transforms(
        cov, np.array([1.0, 1.0]), np.array([1.0, 1.0]), state
    )
    np.testing.assert_array_equal(adj, cov)


def test_adjust_covariance_no_state_or_empty() -> None:
    cov = np.array([[1.0]])
    assert tr.adjust_covariance_for_transforms(cov, cov, cov, None) is cov
    empty = np.array([])
    out = tr.adjust_covariance_for_transforms(empty, empty, empty, {"gamma_log_idx": 0})
    assert out.size == 0


# ---------------------------------------------------------------------------
# Model / stratified function wrapping
# ---------------------------------------------------------------------------


def test_wrap_model_no_state_returns_original() -> None:
    def model(_xdata: np.ndarray, *p: float) -> np.ndarray:
        return np.asarray(p)

    assert tr.wrap_model_function_with_transforms(model, None) is model
    assert tr.wrap_model_function_with_transforms(model, {}) is model


def test_wrap_model_noncallable_returns_original() -> None:
    sentinel = object()
    assert tr.wrap_model_function_with_transforms(sentinel, {"gamma_log_idx": 0}) is sentinel


def test_wrap_model_applies_inverse_and_preserves_attrs() -> None:
    captured: dict[str, np.ndarray] = {}

    def model(_xdata: np.ndarray, *p: float) -> np.ndarray:
        captured["physical"] = np.asarray(p)
        return np.asarray(p)

    model.n_phi = 4  # type: ignore[attr-defined]
    state = {"gamma_log_idx": 0, "beta_center_idx": None}
    wrapped = tr.wrap_model_function_with_transforms(model, state)

    # Solver-space gamma = log(value); wrapped should exp() it before calling model.
    wrapped(np.zeros(1), np.log(50.0), 2.0)
    assert captured["physical"][0] == pytest.approx(50.0)
    assert wrapped.n_phi == 4  # attribute preserved


def test_wrap_stratified_no_state_returns_original() -> None:
    def resid(p: np.ndarray) -> np.ndarray:
        return p

    assert tr.wrap_stratified_function_with_transforms(resid, None) is resid


def test_wrap_stratified_applies_inverse_and_delegates_attrs() -> None:
    class Base:
        n_angles = 8

        def __call__(self, p: np.ndarray) -> np.ndarray:
            return p

    state = {"gamma_log_idx": 0, "beta_center_idx": None}
    wrapped = tr.wrap_stratified_function_with_transforms(Base(), state)
    out = wrapped(np.array([np.log(25.0), 1.0]))
    assert out[0] == pytest.approx(25.0)
    # __getattr__ delegates to the wrapped base object.
    assert wrapped.n_angles == 8
