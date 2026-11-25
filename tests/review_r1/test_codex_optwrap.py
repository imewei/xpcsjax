"""Red tests for Codex review shard ``optwrap`` (round 1). One test per claim."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.optimization.test_fixed_parameters_integration import (
    _ALL_PHYSICAL_NAMES,
    _config,
    _synthetic_data,
)
from xpcsjax.config import ConfigManager
from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.optimization.nlsq.core import fit_nlsq_jax
from xpcsjax.optimization.nlsq.wrapper import NLSQWrapper

pytestmark = [
    pytest.mark.filterwarnings("ignore:Ill-conditioned Jacobian:UserWarning"),
]


def test_optwrap_b1():
    w = NLSQWrapper()
    t = np.arange(1, 9) * 0.1
    d = SimpleNamespace(
        phi=np.array([0.0, 90.0]),
        t1=t,
        t2=t,
        g2=np.zeros((2, 8, 8)),
        sigma=None,
        q=0.01,
        L=2e6,
        dt=0.1,
    )
    p = [0.3, 1.0, 1000.0, 0.5, 10.0, 0.1, 0.0, 0.0, 0.0]
    x = np.arange(d.g2.size)
    mode = AnalysisMode.LAMINAR_FLOW
    y = np.asarray(w._create_residual_function(d, mode, True, "averaged")(x, *p))
    y = y.reshape(2, -1)
    assert not np.allclose(y[0], y[1])  # the two angles genuinely differ
    d.phi, d.g2 = d.phi[::-1], d.g2[::-1]
    z = np.asarray(w._create_residual_function(d, mode, True, "averaged")(x, *p))
    np.testing.assert_allclose(z.reshape(2, -1), y[::-1])


def test_optwrap_b2():
    d = SimpleNamespace(
        phi=np.array([0.0, 90.0]),
        phi_raw=np.array([90.0, 0.0]),
        phi_flat=np.array([90.0, 0.0]),
        t1=np.array([0.0]),
        t2=np.array([1.0]),
        t1_flat=np.zeros(2),
        t2_flat=np.ones(2),
        sigma=np.array([[[0.1]], [[1.0]]]),
    )
    np.testing.assert_allclose(NLSQWrapper._prepare_sigma_data(None, d, 2), [0.1, 1.0])


def test_optwrap_b3():
    cfg = ConfigManager(
        config_override=_config(
            active_parameters=[p for p in _ALL_PHYSICAL_NAMES if p != "D_offset"],
            extra_top={"optimization": {"stratification": {"force_sequential_fallback": True}}},
        )
    )
    r = fit_nlsq_jax(_synthetic_data(), cfg, use_adapter=False)
    assert r.device_info["strategy"] == "sequential_per_angle"


def test_optwrap_b5():
    data = _synthetic_data("laminar_flow")
    cm = ConfigManager(config_override=_config("laminar_flow", fixed_parameters={"D_offset": 37.5}))
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    params = np.asarray(result.parameters).ravel()
    assert np.isfinite(result.chi_squared) and result.chi_squared > 0
    assert result.reduced_chi_squared == pytest.approx(
        result.chi_squared / (data["g2"].size - (len(params) - 1))
    )


def test_optwrap_b7():
    from xpcsjax.optimization.nlsq.adapter import NLSQAdapter

    t = np.arange(6.0)
    data = {"phi": np.array([0.0, 45.0, 90.0]), "t1": t, "t2": t, "g2": np.zeros((3, 6, 6))}
    x, y, _ = NLSQAdapter()._flatten_xpcs_data(data)
    assert x.shape == (y.size, 3)


def test_optwrap_b8():
    import jax.numpy as jnp

    from tests.optimization.test_fixed_parameters_integration import (
        CONTRAST,
        DT,
        OFFSET,
        TRUE_PHYSICAL_LAMINAR,
        L,
        Q,
    )
    from xpcsjax.core.jax_backend import compute_g2_scaled
    from xpcsjax.optimization.nlsq.adapter import NLSQAdapter

    phys = list(TRUE_PHYSICAL_LAMINAR.values())
    phys[3] = 0.1  # gamma_dot_t0: strong enough shear that the gap is unmistakable
    t = np.arange(1, 9) * DT
    t1, t2 = np.meshgrid(t, t, indexing="ij")

    def truth(gap):
        return float(
            np.asarray(
                compute_g2_scaled(
                    jnp.array(phys),
                    jnp.asarray(t1),
                    jnp.asarray(t2),
                    jnp.asarray(0.0),
                    Q,
                    gap,
                    CONTRAST,
                    OFFSET,
                    DT,
                )
            )[0][0, 7]
        )

    assert abs(truth(L) - truth(1.0)) > 1e-4  # the gap genuinely matters here
    data = {"phi": np.array([0.0]), "q": Q, "L": L, "dt": DT}
    f, *_ = NLSQAdapter()._build_model_function(data, {}, AnalysisMode.LAMINAR_FLOW, True, 1)
    got = float(np.asarray(f(np.array([[t[0], t[7], 0.0]]), CONTRAST, OFFSET, *phys))[0])
    assert got == pytest.approx(truth(L), rel=1e-6)


def test_optwrap_b9(monkeypatch):
    from tests.optimization.test_cmaes_trigger import _laminar_cmaes_config, _tiny_laminar_data
    from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESResult, CMAESWrapper
    from xpcsjax.optimization.nlsq.core import fit_nlsq_cmaes

    d = _tiny_laminar_data(n_phi=3)
    # Distinct baseline per angle, then present the angles in DESCENDING order.
    d["c2_exp"] = d["c2_exp"] + np.array([0.0, 0.1, 0.2])[:, None, None]
    for key in ("phi_angles_list", "c2_exp"):
        d[key] = d[key][::-1].copy()
    cfg = _laminar_cmaes_config("constant")
    cfg.config["optimization"]["nlsq"]["cmaes"]["nlsq_warmstart"] = False
    seen = {}

    def fit(self, model_func, xdata, p0, **kw):  # capture the model only
        seen["y"] = np.asarray(model_func(xdata, *p0)).reshape(3, -1)
        return CMAESResult(np.asarray(p0), None, 1.0, True)

    monkeypatch.setattr(CMAESWrapper, "fit", fit)
    fit_nlsq_cmaes(d, cfg)
    data_rows = d["c2_exp"].reshape(3, -1).mean(axis=1)  # descending baselines
    model_rows = seen["y"].mean(axis=1)
    # The frozen per-angle scaling must follow the data rows it was estimated from.
    np.testing.assert_array_equal(np.argsort(model_rows), np.argsort(data_rows))


def test_optwrap_b10(monkeypatch):
    from xpcsjax.optimization.nlsq import wrapper

    actual = wrapper.select_nlsq_strategy
    seen = []

    def select(n_points, n_params, memory_fraction=0.75, **kw):
        seen.append(memory_fraction)
        return actual(n_points, n_params, memory_fraction, **kw)

    monkeypatch.setattr(wrapper, "select_nlsq_strategy", select)
    cfg = ConfigManager(
        config_override=_config(
            extra_top={"optimization": {"nlsq": {"memory_fraction": 0.25}}},
        )
    )
    fit_nlsq_jax(_synthetic_data(), cfg, use_adapter=False)
    assert seen and all(fraction == 0.25 for fraction in seen)
