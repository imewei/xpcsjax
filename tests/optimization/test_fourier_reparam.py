"""Scientific tests for xpcsjax.optimization.nlsq.fourier_reparam.

Fourier reparameterization maps per-angle contrast/offset to truncated Fourier
coefficients via a basis matrix B (``per_angle = B @ coeffs``). The central
correctness property is **exact round-trip for representable signals**: a
per-angle array that IS a low-order Fourier series must recover its
coefficients (least-squares) and reconstruct identically. Independent mode is a
plain concatenate/split, also round-trip exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.optimization.nlsq import fourier_reparam as fr

# 23 angles over a full period — enough for order-2 Fourier.
_PHI = np.linspace(-np.pi, np.pi, 23, endpoint=False)


def _fourier(order: int = 2, mode: str = "fourier") -> fr.FourierReparameterizer:
    cfg = fr.FourierReparamConfig(mode=mode, fourier_order=order)  # type: ignore[arg-type]
    return fr.FourierReparameterizer(_PHI, cfg)


def _basis(phi: np.ndarray, order: int) -> np.ndarray:
    cols = [np.ones_like(phi)]
    for k in range(1, order + 1):
        cols.append(np.cos(k * phi))
        cols.append(np.sin(k * phi))
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# config + mode determination
# ---------------------------------------------------------------------------


def test_config_from_dict() -> None:
    cfg = fr.FourierReparamConfig.from_dict(
        {"per_angle_mode": "fourier", "fourier_order": 3, "fourier_auto_threshold": 4}
    )
    assert cfg.mode == "fourier"
    assert cfg.fourier_order == 3
    assert cfg.auto_threshold == 4


def test_mode_fourier_active_for_enough_angles() -> None:
    f = _fourier(order=2, mode="fourier")
    assert f.use_fourier is True
    assert f.n_coeffs_per_param == 5  # 1 + 2*order
    assert f.n_coeffs == 10


def test_mode_fourier_falls_back_when_too_few_angles() -> None:
    cfg = fr.FourierReparamConfig(mode="fourier", fourier_order=2)
    f = fr.FourierReparameterizer(np.array([0.0, 1.0]), cfg)  # n_phi=2 < 5
    assert f.use_fourier is False
    assert f.n_coeffs == 4  # 2 * n_phi


def test_mode_independent() -> None:
    f = _fourier(mode="independent")
    assert f.use_fourier is False
    assert f.n_coeffs == 2 * 23


@pytest.mark.parametrize(
    ("n_phi", "threshold", "expected"),
    [(10, 6, True), (5, 6, False), (7, 6, True)],
)
def test_mode_auto(n_phi: int, threshold: int, expected: bool) -> None:
    cfg = fr.FourierReparamConfig(mode="auto", fourier_order=2, auto_threshold=threshold)
    f = fr.FourierReparameterizer(np.linspace(0, np.pi, n_phi), cfg)
    assert f.use_fourier is expected


# ---------------------------------------------------------------------------
# round-trip identity (the core scientific invariant)
# ---------------------------------------------------------------------------


def test_roundtrip_exact_for_representable_signal() -> None:
    f = _fourier(order=2)
    B = _basis(_PHI, 2)
    c_true = np.array([0.3, 0.05, -0.02, 0.01, 0.0])
    o_true = np.array([1.0, 0.1, 0.0, -0.05, 0.02])
    contrast = B @ c_true
    offset = B @ o_true

    coeffs = f.per_angle_to_fourier(contrast, offset)
    # Recovered coefficients match the generating coefficients.
    np.testing.assert_allclose(coeffs[:5], c_true, atol=1e-10)
    np.testing.assert_allclose(coeffs[5:], o_true, atol=1e-10)

    # And reconstruction is exact.
    c_back, o_back = f.fourier_to_per_angle(coeffs)
    np.testing.assert_allclose(c_back, contrast, atol=1e-10)
    np.testing.assert_allclose(o_back, offset, atol=1e-10)


def test_roundtrip_independent_mode_is_exact() -> None:
    cfg = fr.FourierReparamConfig(mode="independent")
    phi = np.linspace(0, np.pi, 4)
    f = fr.FourierReparameterizer(phi, cfg)
    contrast = np.array([0.1, 0.2, 0.3, 0.4])
    offset = np.array([1.0, 1.1, 1.2, 1.3])
    coeffs = f.per_angle_to_fourier(contrast, offset)
    np.testing.assert_array_equal(coeffs, np.concatenate([contrast, offset]))
    c_back, o_back = f.fourier_to_per_angle(coeffs)
    np.testing.assert_array_equal(c_back, contrast)
    np.testing.assert_array_equal(o_back, offset)


def test_single_group_to_from_fourier_roundtrip() -> None:
    f = _fourier(order=2)
    B = _basis(_PHI, 2)
    coeffs_true = np.array([0.5, 0.1, -0.1, 0.05, 0.0])
    values = B @ coeffs_true
    recovered = f.to_fourier(values)
    np.testing.assert_allclose(recovered, coeffs_true, atol=1e-10)
    np.testing.assert_allclose(f.from_fourier(recovered), values, atol=1e-10)


def test_single_group_independent_mode_passthrough() -> None:
    f = _fourier(mode="independent")
    vals = np.arange(23.0)
    np.testing.assert_array_equal(f.to_fourier(vals), vals)
    np.testing.assert_array_equal(f.from_fourier(vals), vals)


# ---------------------------------------------------------------------------
# basis / jacobian / order
# ---------------------------------------------------------------------------


def test_get_basis_matrix() -> None:
    f = _fourier(order=2)
    B = f.get_basis_matrix()
    assert B is not None
    assert B.shape == (23, 5)
    np.testing.assert_allclose(B[:, 0], 1.0)  # constant term
    np.testing.assert_allclose(B[:, 1], np.cos(_PHI))
    assert _fourier(mode="independent").get_basis_matrix() is None


def test_order_property() -> None:
    assert _fourier(order=3).order == 3


def test_jacobian_transform_fourier_blocks() -> None:
    f = _fourier(order=2)
    J = f.get_jacobian_transform()
    assert J.shape == (2 * 23, 10)
    B = f.get_basis_matrix()
    assert B is not None
    np.testing.assert_allclose(J[:23, :5], B)  # contrast block
    np.testing.assert_allclose(J[23:, 5:], B)  # offset block
    np.testing.assert_allclose(J[:23, 5:], 0.0)  # off-diagonal zero


def test_jacobian_transform_independent_is_identity() -> None:
    f = _fourier(mode="independent")
    J = f.get_jacobian_transform()
    np.testing.assert_array_equal(J, np.eye(2 * 23))


# ---------------------------------------------------------------------------
# bounds / initial coeffs / labels / diagnostics
# ---------------------------------------------------------------------------


def test_get_bounds_fourier() -> None:
    cfg = fr.FourierReparamConfig(
        mode="fourier",
        fourier_order=2,
        c0_bounds=(0.1, 0.8),
        ck_bounds=(-0.2, 0.2),
        o0_bounds=(0.5, 1.5),
        ok_bounds=(-0.3, 0.3),
    )
    f = fr.FourierReparameterizer(_PHI, cfg)
    lower, upper = f.get_bounds()
    assert lower.shape == (10,)
    assert (lower[0], upper[0]) == (0.1, 0.8)  # c0
    assert (lower[1], upper[1]) == (-0.2, 0.2)  # c1 harmonic
    assert (lower[5], upper[5]) == (0.5, 1.5)  # o0
    assert (lower[6], upper[6]) == (-0.3, 0.3)  # o1 harmonic


def test_get_bounds_independent() -> None:
    f = _fourier(mode="independent")
    lower, upper = f.get_bounds()
    assert lower.shape == (2 * 23,)


def test_get_initial_coefficients_scalar_and_array() -> None:
    f = _fourier(order=2)
    coeffs = f.get_initial_coefficients(0.3, 1.0)
    # Uniform contrast 0.3 -> c0 == 0.3, harmonics ~ 0.
    np.testing.assert_allclose(coeffs[0], 0.3, atol=1e-10)
    np.testing.assert_allclose(coeffs[1:5], 0.0, atol=1e-10)
    np.testing.assert_allclose(coeffs[5], 1.0, atol=1e-10)
    # Array input is also accepted.
    arr_coeffs = f.get_initial_coefficients(np.full(23, 0.4), np.full(23, 1.2))
    np.testing.assert_allclose(arr_coeffs[0], 0.4, atol=1e-10)


def test_coefficient_labels_fourier() -> None:
    labels = _fourier(order=2).get_coefficient_labels()
    assert labels[:5] == ["contrast_c0", "contrast_c1", "contrast_s1", "contrast_c2", "contrast_s2"]
    assert "offset_c0" in labels


def test_coefficient_labels_independent() -> None:
    labels = _fourier(mode="independent").get_coefficient_labels()
    assert labels[0] == "contrast[0]"
    assert labels[23] == "offset[0]"


def test_get_diagnostics() -> None:
    diag = _fourier(order=2).get_diagnostics()
    assert diag["use_fourier"] is True
    assert diag["n_coeffs"] == 10
    assert diag["reduction_ratio"] == pytest.approx(10 / (2 * 23))


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def test_fourier_to_per_angle_validation() -> None:
    f = _fourier(order=2)
    with pytest.raises(ValueError, match="must be 1D"):
        f.fourier_to_per_angle(np.zeros((2, 5)))
    with pytest.raises(ValueError, match="Expected 10"):
        f.fourier_to_per_angle(np.zeros(3))


def test_per_angle_to_fourier_validation() -> None:
    f = _fourier(order=2)
    with pytest.raises(ValueError, match="contrast must be 1D"):
        f.per_angle_to_fourier(np.zeros((2, 2)), np.zeros(23))
    with pytest.raises(ValueError, match="Expected 23 contrast"):
        f.per_angle_to_fourier(np.zeros(5), np.zeros(23))
    with pytest.raises(ValueError, match="Expected 23 offset"):
        f.per_angle_to_fourier(np.zeros(23), np.zeros(5))


def test_to_from_fourier_validation() -> None:
    f = _fourier(order=2)
    with pytest.raises(ValueError, match="must be 1D"):
        f.to_fourier(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="Expected 23 values"):
        f.to_fourier(np.zeros(5))
    with pytest.raises(ValueError, match="Expected 5 coefficients"):
        f.from_fourier(np.zeros(3))


# ---------------------------------------------------------------------------
# model wrapper
# ---------------------------------------------------------------------------


def test_create_fourier_model_wrapper() -> None:
    f = _fourier(order=2)
    n_physical = 3
    captured: dict[str, np.ndarray] = {}

    def model_fn(full_params: np.ndarray, x: np.ndarray) -> np.ndarray:
        captured["full"] = full_params
        return full_params

    wrapped = fr.create_fourier_model_wrapper(model_fn, f, n_physical)
    params = np.concatenate([f.get_initial_coefficients(0.3, 1.0), np.array([1.0, 2.0, 3.0])])
    out = wrapped(params, np.zeros(1))
    # Full param vector = contrast(23) + offset(23) + physical(3).
    assert out.shape == (2 * 23 + 3,)
    np.testing.assert_array_equal(captured["full"][-3:], [1.0, 2.0, 3.0])
