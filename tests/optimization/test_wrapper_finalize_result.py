"""PR #37 review: the standard in-memory NLSQ path's chi_squared/pcov fixes had
zero regression coverage that runs in default CI.

``_finalize_result`` in ``wrapper.py`` (the standard, most-common fit path)
previously computed ``chi_squared`` from the raw MODEL output instead of true
residuals ``(ydata - model)`` — a silent wrong-quality-metric bug on the
default path. The only guard was ``tests/parity/test_homodyne_engine_preservation
.py::test_laminar_flow_end_to_end_golden``, whose ``rtol=1e-10`` value-compare
only runs under ``XPCSJAX_RUN_ENGINE_PARITY=1`` (a maintainer-local, pre-release
gate per CLAUDE.md — not part of ``make test``/``make verify``/CI).

These tests don't need bit-exact goldens: a small synthetic fit with known,
tiny noise makes the two candidate quantities (true SSR vs raw model-value
sum-of-squares) differ by roughly two orders of magnitude, so a coarse
sentinel bound is enough to catch a regression back to the bug without being
fragile to legitimate numeric drift.
"""

from __future__ import annotations

import numpy as np

_PHYS_NAMES = [
    "D0",
    "alpha",
    "D_offset",
    "gamma_dot_t0",
    "beta",
    "gamma_dot_t_offset",
    "phi0",
]


def _build_laminar_config(diagnostics_enabled: bool = False):
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.homodyne_model import HomodyneModel

    n_t = 8
    phi = np.array([0.0, 90.0], dtype=np.float64)
    t = np.linspace(0.0, float(n_t - 1), n_t, dtype=np.float64)
    true_params = np.array([1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0], dtype=np.float64)

    nlsq_settings = {
        "analysis_mode": "laminar_flow",
        "max_iterations": 50,
        "loss": "linear",
        "cmaes": {"enable": False, "auto_select": False},
        "multi_start": {"enable": False},
        "anti_degeneracy": {
            "enable": True,
            "per_angle_mode": "individual",
            "constant_scaling_threshold": 3,
        },
    }
    if diagnostics_enabled:
        nlsq_settings["diagnostics"] = {"enable": True, "sample_size": 512}

    config_dict = {
        "analysis_mode": "laminar_flow",
        "analyzer_parameters": {
            "dt": 0.1,
            "start_frame": 1,
            "end_frame": n_t,
            "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": n_t},
            "scattering": {"wavevector_q": 0.0237},
            "geometry": {"stator_rotor_gap": 2000000},
        },
        "initial_parameters": {
            "parameter_names": list(_PHYS_NAMES),
            "values": true_params.tolist(),
        },
        "optimization": {
            "method": "nlsq",
            "nlsq": nlsq_settings,
            "stratification": {"enabled": False},
        },
    }

    cfg = ConfigManager(config_override=config_dict)
    model = HomodyneModel(cfg.config)
    c2 = np.asarray(
        model.compute_c2(true_params, phi, contrast=0.3, offset=1.0),
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed=20260806)
    c2 = c2 + rng.normal(0.0, 5e-4, size=c2.shape)

    data = {
        "phi_angles_list": phi,
        "c2_exp": c2,
        "t1": t,
        "t2": t,
        "wavevector_q_list": np.array([0.0237], dtype=np.float64),
    }
    return data, cfg


def test_standard_path_chi_squared_is_from_true_residuals_not_model_output():
    """Regression sentinel for the ``_finalize_result`` chi_squared bug.

    With 5e-4 noise and the default sigma=0.01, a correctly-computed
    chi_squared for a converged fit is O(1-10). The reverted bug (summing the
    raw model output squared, values ~O(1) over ~128 points) would land two
    orders of magnitude higher, comfortably outside this bound.
    """
    from xpcsjax.optimization.nlsq import fit_nlsq

    data, cfg = _build_laminar_config()
    result = fit_nlsq(data, cfg)

    assert np.isfinite(result.chi_squared)
    assert result.chi_squared > 0
    assert result.chi_squared < 20.0, (
        f"chi_squared={result.chi_squared} is far too large for a converged "
        "small-noise fit — looks like it was computed from raw model output "
        "instead of true residuals (ydata - model)"
    )
    assert result.reduced_chi_squared < 5.0


def test_post_process_results_never_reassigns_pcov_from_diagnostics():
    """Regression sentinel for the diagnostics-overwrites-pcov bug.

    A live end-to-end fit with ``diagnostics.enable=True`` is not used here —
    it exercises an unrelated pre-existing crash on tiny synthetic fixtures
    (``model_function``'s ``jnp.stack`` on an empty tuple inside the recovery
    path, not touched by this PR). Instead this pins the fix at the source
    level: the diagnostics-only sampled-Jacobian covariance must be stored
    under its own key and ``pcov``/``popt`` must never be reassigned from it,
    per this file's own stated "diagnostics never mutates pcov" contract.
    """
    import inspect

    from xpcsjax.optimization.nlsq import wrapper

    src = inspect.getsource(wrapper.NLSQWrapper._post_process_results)
    assert 'diagnostics_payload["diagnostic_covariance"]' in src, (
        "diagnostics-derived covariance must be stored under its own key"
    )
    # No `pcov = <diagnostics-derived expression>` assignment anywhere from
    # the start of the diagnostics block through where the covariance key is
    # written — pcov must stay solver-derived, never reassigned here.
    diag_block_start = src.index("diagnostics_enabled and diagnostics_sample_x")
    diag_block = src[diag_block_start:]
    before_key_write = diag_block.split('diagnostics_payload["diagnostic_covariance"]')[0]
    assert "pcov = " not in before_key_write
