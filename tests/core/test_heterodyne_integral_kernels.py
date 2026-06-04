"""Characterization tests for the heterodyne velocity/transport integral kernels.

Quality-gate gap: ``compute_velocity_integral_matrix`` and
``compute_transport_integral_matrix`` — the core flow/transport time-integral
kernels — had zero direct tests. These pin their analytic behaviour at the
constant-rate limit and their structural (anti)symmetry, so a regression in the
``trapezoid_cumsum`` → ``create_signed_integral_matrix`` pipeline is caught.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from xpcsjax.core.heterodyne_jax_backend import (
    compute_transport_integral_matrix,
    compute_velocity_integral_matrix,
)

_N = 6
_DT = 0.5
_C = 2.0  # constant rate
_T = jnp.arange(_N) * _DT
_I, _J = np.meshgrid(np.arange(_N), np.arange(_N), indexing="ij")


def test_velocity_integral_constant_velocity_is_linear_signed_gap():
    # v0=0 ⇒ v(t)=v_offset=_C (constant). M[i,j] = ∫_{t_i}^{t_j} v dt' = C·(t_j−t_i).
    m = np.asarray(compute_velocity_integral_matrix(_T, v0=0.0, beta=1.0, v_offset=_C, dt=_DT))
    expected = _C * (_J - _I) * _DT
    assert np.allclose(m, expected, atol=1e-10)


def test_velocity_integral_is_antisymmetric_with_zero_diagonal():
    m = np.asarray(compute_velocity_integral_matrix(_T, v0=1.0, beta=1.0, v_offset=0.3, dt=_DT))
    assert np.allclose(m, -m.T, atol=1e-12)
    assert np.allclose(np.diag(m), 0.0, atol=1e-12)


def test_transport_integral_constant_rate_is_abs_gap_and_symmetric():
    # D0=0 ⇒ J_rate=offset=_C (constant). M[i,j] = |C·(t_j−t_i)| = C·|j−i|·dt.
    m = np.asarray(compute_transport_integral_matrix(_T, D0=0.0, alpha=1.0, offset=_C, dt=_DT))
    expected = _C * np.abs(_J - _I) * _DT
    # smooth_abs is exact away from zero; allow a small tolerance near the diagonal.
    assert np.allclose(m, expected, atol=1e-6)
    assert np.allclose(m, m.T, atol=1e-10)  # |·| ⇒ symmetric
    assert np.all(m >= -1e-9)  # non-negative
