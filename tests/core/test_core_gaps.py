"""Gap-filling tests for xpcsjax.core identified by Codex + Gemini review.

Covers four previously untested paths:
  T2 - HeterodyneModel.compute_g1 multi-phi vmap output shape
  T3 - HeterodyneModel.compute_residual end-to-end flat residual
  T8 - HomodyneModel.__init__ rejects negative end_frame sentinel
  T10 - make_model handles "heterodyne" synonym and non-string analysis_mode
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from xpcsjax.core.heterodyne_model import HeterodyneModel
from xpcsjax.core.models import make_model


# ---------------------------------------------------------------------------
# T2: HeterodyneModel.compute_g1 — multi-phi vmap path
# ---------------------------------------------------------------------------

class TestHeterodyneComputeG1MultiPhi:
    """HeterodyneModel.compute_g1 with a phi array exercises jax.vmap."""

    @pytest.fixture
    def model_and_params(self) -> tuple[HeterodyneModel, jnp.ndarray]:
        model = HeterodyneModel()
        params = model.get_default_parameters()
        return model, params

    def test_scalar_phi_returns_2d(self, model_and_params: tuple) -> None:
        """Scalar phi → (N, N) output."""
        model, params = model_and_params
        N = 8
        t = jnp.linspace(0.0, 7.0, N, dtype=jnp.float64)
        t1, t2 = jnp.meshgrid(t, t, indexing="ij")
        phi_scalar = jnp.float64(0.0)
        q, L, dt = 0.01, 1.0, 1.0

        out = model.compute_g1(params, t1, t2, phi_scalar, q, L, dt)

        assert out.ndim == 2
        assert out.shape == (N, N)
        assert jnp.all(jnp.isfinite(out))

    def test_multi_phi_returns_3d_correct_shape(self, model_and_params: tuple) -> None:
        """1-D phi of length n_phi → (n_phi, N, N) output via vmap."""
        model, params = model_and_params
        N = 8
        n_phi = 3
        t = jnp.linspace(0.0, 7.0, N, dtype=jnp.float64)
        t1, t2 = jnp.meshgrid(t, t, indexing="ij")
        phi_arr = jnp.array([0.0, math.pi / 4, math.pi / 2], dtype=jnp.float64)
        q, L, dt = 0.01, 1.0, 1.0

        out = model.compute_g1(params, t1, t2, phi_arr, q, L, dt)

        assert out.ndim == 3
        assert out.shape == (n_phi, N, N)
        assert jnp.all(jnp.isfinite(out)), "vmap result must be fully finite"

    def test_multi_phi_each_angle_differs(self, model_and_params: tuple) -> None:
        """Two distinct phi values must produce numerically different surfaces.

        The heterodyne c2 kernel depends on phi via the velocity term
        v0*cos(phi). With default params the differences are small but
        non-zero. We verify that the max absolute difference exceeds
        floating-point epsilon, confirming independent vmap evaluations.
        """
        model, params = model_and_params
        N = 8
        t = jnp.linspace(0.0, 7.0, N, dtype=jnp.float64)
        t1, t2 = jnp.meshgrid(t, t, indexing="ij")
        phi_arr = jnp.array([0.0, math.pi / 2], dtype=jnp.float64)
        q, L, dt = 0.05, 1.0, 1.0

        out = model.compute_g1(params, t1, t2, phi_arr, q, L, dt)

        max_diff = float(jnp.max(jnp.abs(out[0] - out[1])))
        assert max_diff > 1e-15, (
            f"phi=0 and phi=pi/2 surfaces should differ by more than eps, got {max_diff}"
        )

    def test_1d_t1_also_works(self, model_and_params: tuple) -> None:
        """1-D t1/t2 input (non-meshgrid) falls back to reshape(-1)."""
        model, params = model_and_params
        N = 6
        t1 = jnp.linspace(0.0, 5.0, N, dtype=jnp.float64)
        t2 = jnp.linspace(0.0, 5.0, N, dtype=jnp.float64)
        phi_scalar = jnp.float64(0.0)
        q, L, dt = 0.01, 1.0, 1.0

        out = model.compute_g1(params, t1, t2, phi_scalar, q, L, dt)

        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# T3: HeterodyneModel.compute_residual — end-to-end flat residual
# ---------------------------------------------------------------------------

class TestHeterodyneComputeResidual:
    """HeterodyneModel.compute_residual returns a flat 1-D residual array."""

    @pytest.fixture
    def model(self) -> HeterodyneModel:
        return HeterodyneModel()

    def _make_data(self, N: int = 6, n_phi: int = 2) -> dict:
        t = np.linspace(0.0, float(N - 1), N)
        q = 0.01
        phi_angles = np.linspace(0.0, math.pi / 2, n_phi)
        c2_exp = np.ones((n_phi, N, N), dtype=np.float64)
        return {
            "t": t,
            "q": q,
            "phi_angles_list": phi_angles,
            "c2_exp": c2_exp,
            "dt": 1.0,
            "contrast": 1.0,
            "offset": 0.0,
        }

    def test_residual_is_1d(self, model: HeterodyneModel) -> None:
        """compute_residual must return a 1-D array."""
        params = model.get_default_parameters()
        data = self._make_data(N=6, n_phi=2)
        residual = model.compute_residual(params, data)
        assert residual.ndim == 1

    def test_residual_length_equals_total_elements(self, model: HeterodyneModel) -> None:
        """Length must be n_phi * N * N."""
        N, n_phi = 6, 2
        params = model.get_default_parameters()
        data = self._make_data(N=N, n_phi=n_phi)
        residual = model.compute_residual(params, data)
        assert residual.shape == (n_phi * N * N,)

    def test_residual_finite_for_default_params(self, model: HeterodyneModel) -> None:
        """Residual against a synthetic all-ones c2 must be finite."""
        params = model.get_default_parameters()
        data = self._make_data()
        residual = model.compute_residual(params, data)
        assert jnp.all(jnp.isfinite(residual))

    def test_residual_zero_for_perfect_fit(self, model: HeterodyneModel) -> None:
        """When c2_exp == c2_model, residual must be all-zero."""
        N = 6
        params = model.get_default_parameters()
        t = jnp.linspace(0.0, float(N - 1), N, dtype=jnp.float64)
        t1_grid, t2_grid = jnp.meshgrid(t, t, indexing="ij")
        phi_arr = jnp.array([0.0], dtype=jnp.float64)
        q, dt = 0.01, 1.0

        # Generate model output first, then use it as the "experiment"
        c2_model = model.compute_g1(params, t1_grid, t2_grid, phi_arr, q, 1.0, dt)
        data = {
            "t": np.asarray(t),
            "q": q,
            "phi_angles_list": np.array([0.0]),
            "c2_exp": np.asarray(c2_model),
            "dt": dt,
            "contrast": 1.0,
            "offset": 0.0,
        }
        residual = model.compute_residual(params, data)
        assert jnp.allclose(residual, 0.0, atol=1e-10), (
            "perfect-fit residual must be zero"
        )

    def test_residual_uses_phi_angle_fallback_key(self, model: HeterodyneModel) -> None:
        """compute_residual accepts 'phi_angle' (singular) as well as 'phi_angles_list'."""
        N = 4
        params = model.get_default_parameters()
        t = np.linspace(0.0, float(N - 1), N)
        data = {
            "t": t,
            "q": 0.01,
            "phi_angle": np.float64(0.0),  # singular key
            "c2_exp": np.ones((N, N), dtype=np.float64),
            "dt": 1.0,
        }
        residual = model.compute_residual(params, data)
        assert residual.ndim == 1
        assert residual.shape == (N * N,)


# ---------------------------------------------------------------------------
# T8: HomodyneModel.__init__ — negative end_frame sentinel
# ---------------------------------------------------------------------------

class TestHomodyneModelInitValidation:
    """HomodyneModel must reject negative end_frame before constructing."""

    def _minimal_config(self, end_frame: int = 99) -> dict:
        return {
            "analyzer_parameters": {
                "temporal": {
                    "dt": 1e-3,
                    "start_frame": 0,
                    "end_frame": end_frame,
                },
                "scattering": {"wavevector_q": 0.01},
                "geometry": {"stator_rotor_gap": 1e6},
            }
        }

    def test_negative_end_frame_raises_value_error(self) -> None:
        """end_frame=-1 must raise ValueError before any JAX computation."""
        from xpcsjax.core.homodyne_model import HomodyneModel

        with pytest.raises(ValueError, match="end_frame"):
            HomodyneModel(self._minimal_config(end_frame=-1))

    def test_negative_end_frame_error_mentions_sentinel(self) -> None:
        """Error message must reference 'sentinel' so callers understand the fix."""
        from xpcsjax.core.homodyne_model import HomodyneModel

        with pytest.raises(ValueError, match="sentinel"):
            HomodyneModel(self._minimal_config(end_frame=-99))

    def test_valid_config_constructs_successfully(self) -> None:
        """A properly resolved end_frame must not raise."""
        from xpcsjax.core.homodyne_model import HomodyneModel

        model = HomodyneModel(self._minimal_config(end_frame=99))
        assert model.analysis_mode is not None


# ---------------------------------------------------------------------------
# T10: make_model — "heterodyne" synonym and non-string analysis_mode
# ---------------------------------------------------------------------------

class TestMakeModelDispatch:
    """make_model dispatches correctly for edge-case mode strings."""

    def test_heterodyne_string_returns_heterodyne_model(self) -> None:
        """'heterodyne' synonym must dispatch to HeterodyneModel."""
        model = make_model({"analysis_mode": "heterodyne"})
        assert isinstance(model, HeterodyneModel)

    def test_two_component_string_returns_heterodyne_model(self) -> None:
        """'two_component' must dispatch to HeterodyneModel."""
        model = make_model({"analysis_mode": "two_component"})
        assert isinstance(model, HeterodyneModel)

    def test_two_dash_component_string_returns_heterodyne_model(self) -> None:
        """'two-component' hyphenated variant must dispatch to HeterodyneModel."""
        model = make_model({"analysis_mode": "two-component"})
        assert isinstance(model, HeterodyneModel)

    def test_non_string_mode_raises_value_error(self) -> None:
        """Integer analysis_mode must raise ValueError, not AttributeError."""
        with pytest.raises(ValueError, match="analysis_mode must be a string"):
            make_model({"analysis_mode": 42})

    def test_static_mode_returns_diffusion_or_combined(self) -> None:
        """'static_anisotropic' mode falls through to the homodyne path."""
        from xpcsjax.core.models import PhysicsModelBase

        model = make_model({"analysis_mode": "static_anisotropic"})
        assert isinstance(model, PhysicsModelBase)
        assert not isinstance(model, HeterodyneModel)

    def test_missing_mode_defaults_to_static(self) -> None:
        """Missing analysis_mode key must default to 'static_anisotropic' without error."""
        from xpcsjax.core.models import PhysicsModelBase

        model = make_model({})
        assert isinstance(model, PhysicsModelBase)
