"""Regression tests for JIT-safety of Fourier per-angle conversion.

The heterodyne fourier joint fit traces its residual under ``jax.jit``. The
numpy-based ``FourierReparameterizer.fourier_to_per_angle`` calls ``np.asarray``
on the (traced) coefficient slice, raising ``TracerArrayConversionError`` and
silently degrading the fourier fit. These tests pin the JIT-safe variant and
verify the heterodyne fourier joint path no longer leaks the tracer error.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from xpcsjax.optimization.nlsq.fourier_reparam import (
    FourierReparamConfig,
    FourierReparameterizer,
)


def _make_fourier(n_phi: int = 8, order: int = 2) -> FourierReparameterizer:
    phi = np.linspace(-30.0, 30.0, n_phi).astype(np.float64)
    return FourierReparameterizer(
        np.deg2rad(phi), FourierReparamConfig(mode="fourier", fourier_order=order)
    )


def test_fourier_to_per_angle_jax_matches_numpy_and_is_jit_safe():
    fr = _make_fourier()
    coeffs = np.linspace(0.1, 0.5, fr.n_coeffs).astype(np.float64)
    c_ref, o_ref = fr.fourier_to_per_angle(coeffs)  # concrete numpy reference

    @jax.jit
    def run(x):
        return fr.fourier_to_per_angle_jax(x)

    c_jax, o_jax = run(jnp.asarray(coeffs))  # must trace + run without error
    assert np.max(np.abs(np.asarray(c_jax) - c_ref)) < 1e-12
    assert np.max(np.abs(np.asarray(o_jax) - o_ref)) < 1e-12


def test_heterodyne_fourier_joint_fit_has_no_tracer_error(caplog):
    """The fourier joint fit must not leak TracerArrayConversionError."""
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_jax_backend import compute_c2_heterodyne
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    cfg = ConfigManager("xpcsjax/config/templates/xpcsjax_two_component.yaml")
    model = HeterodyneModel.from_config(cfg.config)
    t = np.arange(1, 13, dtype=np.float64) * model.dt
    model.sync_time_axis(t)

    phi = np.linspace(-30.0, 30.0, 8).astype(np.float64)  # n_phi=8 > fourier_threshold
    full = jnp.asarray(model.param_manager.get_full_values(), dtype=jnp.float64)
    contrast, offset = model.scaling.get_for_angle(0)
    c2 = np.stack(
        [
            np.asarray(compute_c2_heterodyne(full, model.t, model.q, model.dt, float(p), contrast, offset))
            for p in phi
        ]
    )

    nlsq_cfg = NLSQConfig(per_angle_mode="fourier", fourier_order=2, max_iterations=5)
    with caplog.at_level("WARNING"):
        fit_nlsq_multi_phi(model, c2, phi, nlsq_cfg, None)

    leaked = [
        r.getMessage()
        for r in caplog.records
        if "__array__" in r.getMessage() or "TracerArrayConversion" in r.getMessage()
    ]
    assert not leaked, f"tracer error leaked into fourier fit: {leaked[0][:100]}"
