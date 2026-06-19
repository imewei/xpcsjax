"""Task 2 gate: ``on_iteration`` observer threads through the NLSQ engine.

The seam is ``_build_homodyne_l4_callback`` in ``wrapper.py``.  When
``on_iteration`` is not None the function wraps the existing L4 callback so
the observer is called AFTER the L4 work on every iteration; when it IS None
the existing callback object is returned unchanged (byte-identical path).

Fixtures reuse the laminar-flow construction from
``tests/optimization/test_l4_callback_observational.py`` — the smallest
config+data that exercises the live NLSQWrapper STANDARD path.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Shared tiny fixture — mirrors _build_laminar_fit() in
# test_l4_callback_observational.py verbatim (same config dict, same phi/t
# shape, same synthetic-c2 generation).  Kept here so this test module is
# self-contained and does not import from another test.
# ---------------------------------------------------------------------------


def _tiny_static_config_and_data():
    """Return (config, data) for the smallest laminar_flow fit.

    Reuses the exact construction from ``test_l4_callback_observational.py``
    (_build_laminar_fit).  n_t=8, n_phi=2, CMA-ES + multistart + anti-
    degeneracy disabled so the solver takes the NLSQWrapper STANDARD path.
    """
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.homodyne_model import HomodyneModel

    n_t = 8
    phi = np.array([0.0, 90.0], dtype=np.float64)
    t = np.linspace(0.0, float(n_t - 1), n_t, dtype=np.float64)
    true_params = np.array([1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0], dtype=np.float64)

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
            "parameter_names": [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_t0",
                "beta",
                "gamma_dot_t_offset",
                "phi0",
            ],
            "values": true_params.tolist(),
        },
        "optimization": {
            "method": "nlsq",
            "nlsq": {
                "analysis_mode": "laminar_flow",
                "max_iterations": 50,
                "loss": "linear",
                "cmaes": {"enable": False, "auto_select": False},
                "multi_start": {"enable": False},
                "anti_degeneracy": {"enable": False},
            },
            "stratification": {"enabled": False},
        },
    }

    cfg = ConfigManager(config_override=config_dict)

    model = HomodyneModel(cfg.config)
    c2 = np.asarray(
        model.compute_c2(true_params, phi, contrast=0.3, offset=1.0),
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed=20260529)
    c2 = c2 + rng.normal(0.0, 5e-4, size=c2.shape)

    data = {
        "phi_angles_list": phi,
        "c2_exp": c2,
        "t1": t,
        "t2": t,
        "wavevector_q_list": np.array([0.0237], dtype=np.float64),
    }
    return cfg, data


# ---------------------------------------------------------------------------
# Test 1: on_iteration is accepted, completes, and any firings are well-formed
# ---------------------------------------------------------------------------


def test_on_iteration_accepted_and_wellformed():
    """fit_nlsq accepts on_iteration and completes without error.

    Any callback firings must carry a non-negative integer iteration index and
    a positive float SSR.  Iterations must be non-decreasing (monotone index).

    We do NOT hard-require >= 1 firing on this tiny fixture: whether the live
    TRF solve surfaces per-iteration callbacks depends on the fixture/path.
    Guaranteed-positive coverage lives at the service layer (Task 3/4).
    """
    from xpcsjax.optimization.nlsq import fit_nlsq

    cfg, data = _tiny_static_config_and_data()
    seen: list[tuple[int, float]] = []

    fit_nlsq(data, cfg, on_iteration=lambda n, ssr: seen.append((n, ssr)))

    # Every firing must carry (int, positive float)
    assert all(isinstance(n, int) and isinstance(ssr, float) and ssr > 0 for n, ssr in seen), (
        f"Malformed firings: {seen}"
    )
    # Iteration indices must be non-decreasing
    iters = [n for n, _ in seen]
    assert iters == sorted(iters), f"Iterations not monotone: {iters}"


# ---------------------------------------------------------------------------
# Test 2: on_iteration=None is byte-identical to today (parity gate)
# ---------------------------------------------------------------------------


def test_default_none_does_not_change_result():
    """fit_nlsq() with on_iteration=None must be bit-identical to the default.

    This is the PARITY gate: the None branch in _build_homodyne_l4_callback
    returns the UNCHANGED existing callback object so the solve is identical.
    """
    from xpcsjax.optimization.nlsq import fit_nlsq

    cfg, data = _tiny_static_config_and_data()

    r1 = fit_nlsq(data, cfg)
    r2 = fit_nlsq(data, cfg, on_iteration=None)

    np.testing.assert_array_equal(
        np.asarray(r1.parameters),
        np.asarray(r2.parameters),
        err_msg="on_iteration=None changed the result parameters",
    )
    assert r1.chi_squared == r2.chi_squared, (
        f"on_iteration=None changed chi_squared: {r1.chi_squared} vs {r2.chi_squared}"
    )


# ---------------------------------------------------------------------------
# Test 3: observer that raises does NOT abort the fit
# ---------------------------------------------------------------------------


def test_raising_observer_does_not_abort_fit():
    """A raising observer must be silently swallowed; the fit must complete."""
    from xpcsjax.optimization.nlsq import fit_nlsq

    cfg, data = _tiny_static_config_and_data()

    def _bad_observer(n: int, ssr: float) -> None:
        raise RuntimeError("observer intentionally raises")

    # Must not raise
    result = fit_nlsq(data, cfg, on_iteration=_bad_observer)
    # And must still return a valid result
    assert result is not None
    assert result.parameters is not None


# ---------------------------------------------------------------------------
# Test 4: two_component (heterodyne) accepts on_iteration and NEVER calls it
# ---------------------------------------------------------------------------


def test_two_component_accepts_on_iteration_and_never_calls_it():
    """fit_nlsq on two_component config: on_iteration accepted, never called.

    The heterodyne branch ignores on_iteration (live SSR is a follow-up);
    but the public API must not raise TypeError.
    """
    from tests.optimization._heterodyne_fixtures import make_cfgmgr_and_data
    from xpcsjax.optimization.nlsq import fit_nlsq

    cfg, data = make_cfgmgr_and_data(n_phi=3, n_t=12)

    call_count = 0

    def _counting_observer(n: int, ssr: float) -> None:
        nonlocal call_count
        call_count += 1

    # Must not raise
    result = fit_nlsq(data, cfg, on_iteration=_counting_observer)
    assert result is not None
    # Heterodyne ignores the observer — it must NOT be called
    assert call_count == 0, f"two_component unexpectedly called on_iteration {call_count} time(s)"
