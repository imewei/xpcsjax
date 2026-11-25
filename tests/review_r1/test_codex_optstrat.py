"""Red tests for the Codex review of xpcsjax/optimization/nlsq/strategies/ (round 1).

One test per CONFIRMED claim (optstrat-B*). Each is expected to FAIL until the
corresponding defect is fixed.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

_LOG = logging.getLogger("test_codex_optstrat")
_LAMINAR_NAMES = ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]


# --------------------------------------------------------------------------- B1
@pytest.mark.parametrize("engine", ["jit", "plain"])
def test_optstrat_b1(engine: str) -> None:
    """Stratified engines must pair sigma (raw loader phi order) with the right angle."""
    from xpcsjax.optimization.nlsq.strategies.residual import StratifiedResidualFunction
    from xpcsjax.optimization.nlsq.strategies.residual_jit import StratifiedResidualFunctionJIT
    from xpcsjax.optimization.nlsq.strategies.stratified_ls import create_stratified_chunks

    phi_raw = np.array([90.0, 0.0])  # loader order, NOT ascending
    phi = np.repeat(phi_raw, 4)
    t1 = np.tile([1.0, 1.0, 2.0, 2.0], 2)
    t2 = np.tile([1.0, 2.0, 1.0, 2.0], 2)
    g2 = np.full(8, 2.0)
    sigma = np.broadcast_to(np.array([2.0, 4.0])[:, None, None], (2, 2, 2)).copy()
    strat = SimpleNamespace(
        phi_flat=phi,
        t1_flat=t1,
        t2_flat=t2,
        g2_flat=g2,
        sigma=sigma,  # axis 0 follows phi_raw: sigma=2 at 90 deg, sigma=4 at 0 deg
        phi_raw=phi_raw,
        q=0.01,
        L=1.0e6,
        dt=1.0,
        chunk_sizes=[8],
    )
    chunked = create_stratified_chunks(strat)
    cls = StratifiedResidualFunctionJIT if engine == "jit" else StratifiedResidualFunction
    fn = cls(chunked, False, _LAMINAR_NAMES)
    # scalar contrast=0, offset=1 -> theory == 1 everywhere -> raw residual == 1.
    params = np.array([0.0, 1.0, 1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0])
    r = np.asarray(fn(params))
    # index 1: phi=90 (sigma 2), index 5: phi=0 (sigma 4); both off-diagonal.
    np.testing.assert_allclose(r[[1, 5]], [0.5, 0.25])


# --------------------------------------------------------------------------- B2
def test_optstrat_b2() -> None:
    """Heterodyne streaming sigma must follow its angle when phi is descending."""
    from tests.optimization.test_heterodyne_hybrid_streaming import _make_synthetic_heterodyne
    from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
        build_heterodyne_stratified_data,
    )
    from xpcsjax.optimization.nlsq.strategies.heterodyne_hybrid_streaming import (
        build_heterodyne_pointwise_model,
    )

    model, c2, phi = _make_synthetic_heterodyne(n_phi=2, n_t=8)
    phi_desc, c2_desc = phi[::-1].copy(), c2[::-1].copy()  # [+5, -5]
    weights = np.broadcast_to(np.array([4.0, 25.0])[:, None, None], c2.shape).copy()
    strat = build_heterodyne_stratified_data(model, c2_desc, phi_desc, weights=weights)
    _fn, x_data, _y, _p0, meta = build_heterodyne_pointwise_model(
        stratified_data=strat,
        model=model,
        physical_param_names=list(model.param_manager.varying_names),
    )
    # x_data[:, 0] is the SORTED phi index: 1 == +5 deg (weight 4 -> sigma 0.5),
    # 0 == -5 deg (weight 25 -> sigma 0.2).
    np.testing.assert_allclose(meta["sigma"], np.where(x_data[:, 0] == 1, 0.5, 0.2))


# --------------------------------------------------------------------------- B3
class _Reached(BaseException):
    """Raised by the solver spy; BaseException so no ``except Exception`` eats it."""


def test_optstrat_b3(monkeypatch: pytest.MonkeyPatch) -> None:
    """A legal zero-contrast start must reach the solver, not die in the preflight."""
    from tests.parity.test_laminar_execute_layers import _build_laminar_stratified_data
    from xpcsjax.optimization.nlsq.strategies import stratified_ls

    class _SpyLS:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def least_squares(self, *a: Any, **k: Any) -> Any:
            raise _Reached

    monkeypatch.setattr(stratified_ls, "LeastSquares", _SpyLS)
    strat, x0, bounds, n_phi = _build_laminar_stratified_data()
    x0 = x0.copy()
    x0[:n_phi] = 0.0  # every contrast on its legal lower bound
    with pytest.raises(_Reached):
        stratified_ls.fit_with_stratified_least_squares(
            strat,
            True,
            _LAMINAR_NAMES,
            x0,
            bounds,
            _LOG,
            target_chunk_size=2000,
            anti_degeneracy_config={"per_angle_mode": "individual"},
        )


# --------------------------------------------------------------------------- B4
@pytest.mark.filterwarnings("ignore:Ill-conditioned Jacobian")
def test_optstrat_b4(monkeypatch: pytest.MonkeyPatch) -> None:
    """GN covariance DOF must count only rows that contribute to the objective."""
    from tests.parity.test_laminar_execute_layers import _fit
    from xpcsjax.optimization.nlsq.strategies import stratified_ls

    seen: dict[str, int] = {}

    def _spy(f: Any, J: Any, *, n_valid: int, n_params: int, loss: str = "linear") -> Any:
        seen["n_valid"] = int(n_valid)
        return np.eye(np.asarray(J).shape[1])

    monkeypatch.setattr(stratified_ls, "gauss_newton_covariance", _spy)
    _fit(False)
    # 5 angles x 12 frames: the t1 == t2 diagonal is masked to a zero residual.
    assert seen["n_valid"] == 5 * 12 * 11


# --------------------------------------------------------------------- OOC probe
def _ooc_probe(
    monkeypatch: pytest.MonkeyPatch,
    *,
    n_t: int,
    p0: np.ndarray,
    accum: Any,
    cost: Any,
    config: dict | None = None,
    resolved_physical: Any = None,
    names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    from xpcsjax.optimization.nlsq import parallel_accumulator
    from xpcsjax.optimization.nlsq.strategies.out_of_core import (
        fit_with_out_of_core_accumulation,
    )

    def _fake_kernels(**_k: Any) -> tuple[Any, Any]:
        def _acc(p: Any, *_a: Any) -> Any:
            return accum(np.asarray(p))

        def _chi2(p: Any, *_a: Any) -> Any:
            return cost(np.asarray(p))

        return _acc, _chi2

    monkeypatch.setattr(parallel_accumulator, "create_ooc_kernels", _fake_kernels)
    t = np.arange(float(n_t))
    data = SimpleNamespace(
        phi=np.array([0.0]),
        t1=t,
        t2=t,
        g2=np.ones((1, n_t, n_t)),
        sigma=None,
        q=0.01,
        L=1e6,
        dt=1.0,
    )
    return fit_with_out_of_core_accumulation(
        None,
        data,
        False,
        names or ["D0"],
        p0,
        None,
        _LOG,
        config or {},
        resolved_physical=resolved_physical,
    )


# --------------------------------------------------------------------------- B5
def test_optstrat_b5(monkeypatch: pytest.MonkeyPatch) -> None:
    """OOC s^2 must use contributing rows (off-diagonal) and FREE parameters only."""
    from xpcsjax.optimization.nlsq.parameter_utils import ResolvedPhysicalParameters

    resolved = ResolvedPhysicalParameters(
        physical_names=["D0", "alpha", "D_offset"],
        values_full=np.array([1.0, 0.5, 1.0]),
        lower_full=np.zeros(3),
        upper_full=np.full(3, 10.0),
        free_mask=np.array([True, False, True]),
    )
    _popt, pcov, _info = _ooc_probe(
        monkeypatch,
        n_t=4,
        p0=np.ones(5),
        accum=lambda p: (np.eye(5), np.zeros(5), 8.0),
        cost=lambda p: 8.0,
        resolved_physical=resolved,
        names=["D0", "alpha", "D_offset"],
    )
    # 12 off-diagonal rows, 4 free parameters -> s^2 = 8 / (12 - 4) = 1.
    assert pcov[0, 0] == pytest.approx(1.0)


# -------------------------------------------------------------------------- B14
def test_optstrat_b14(monkeypatch: pytest.MonkeyPatch) -> None:
    """A start that is already the exact optimum is 'converged', not a stall."""
    _popt, _pcov, info = _ooc_probe(
        monkeypatch,
        n_t=3,
        p0=np.ones(3),
        accum=lambda p: (np.eye(3), np.zeros(3), 0.0),
        cost=lambda p: 0.0,
    )
    assert info["convergence_status"] == "converged"


# --------------------------------------------------------------------------- B6
@pytest.mark.parametrize("weighting", ["uniform", "n_points"])
def test_optstrat_b6(weighting: str) -> None:
    """Non-default weightings must not average in slots an angle never fitted."""
    from xpcsjax.optimization.nlsq.strategies.sequential import combine_angle_results

    def _res(params: list[float], var: list[float]) -> dict[str, Any]:
        return {
            "success": True,
            "parameters": np.array(params),
            "covariance": np.diag(var),
            "cost": 1.0,
            "n_points": 10,
        }

    # Angle A fits only contrast_A (slot 0); angle B only contrast_B (slot 1).
    # The other slot is the untouched initial guess (0.2) with unknown variance.
    params, cov, _cost = combine_angle_results(
        [_res([0.4, 0.2], [0.01, np.nan]), _res([0.2, 0.8], [np.nan, 0.01])],
        weighting=weighting,
    )
    np.testing.assert_allclose(params, [0.4, 0.8])
    np.testing.assert_allclose(np.diag(cov), [0.01, 0.01])


# --------------------------------------------------------------------------- B8
def test_optstrat_b8(monkeypatch: pytest.MonkeyPatch) -> None:
    """An indefinite Hessian (positive inverse diagonal) must be a placeholder."""
    import jax.numpy as jnp

    from tests.optimization.test_heterodyne_hybrid_streaming import _make_synthetic_heterodyne
    from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
        build_heterodyne_stratified_data,
    )
    from xpcsjax.optimization.nlsq.strategies import heterodyne_hybrid_streaming as hs

    def _indefinite_hessian(fn: Any) -> Any:
        def _h(p: Any) -> Any:
            n = p.shape[0]
            # eigenvalues of the leading block: +1 and -3; inverse diag = +1/3.
            return jnp.eye(n).at[:2, :2].set(jnp.array([[-1.0, 2.0], [2.0, -1.0]]))

        return _h

    monkeypatch.setattr(hs.jax, "hessian", _indefinite_hessian)

    # The L2 solve itself is irrelevant to the covariance guard under test and
    # costs ~45 s of jaxopt compile; return the start point as the "solution".
    from xpcsjax.optimization.nlsq import hierarchical

    monkeypatch.setattr(
        hierarchical.HierarchicalOptimizer,
        "fit",
        lambda self, *, p0, **_k: hierarchical.HierarchicalResult(
            x=np.asarray(p0), fun=1.0, success=True, n_outer_iterations=1
        ),
    )
    model, c2, phi = _make_synthetic_heterodyne(n_phi=2, n_t=5)
    rng = np.random.default_rng(seed=20260822)
    c2_noisy = c2 + rng.normal(0.0, 1e-3, size=c2.shape)
    strat = build_heterodyne_stratified_data(model, c2_noisy, phi, weights=None)
    lo, hi = model.param_manager.get_bounds()
    _popt, _pcov, info = hs.fit_with_stratified_hybrid_streaming_heterodyne(
        stratified_data=strat,
        model=model,
        physical_param_names=list(model.param_manager.varying_names),
        initial_params=np.asarray(model.param_manager.get_initial_values(), dtype=np.float64),
        bounds=(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)),
        hybrid_config={
            "warmup_iterations": 1,
            "max_warmup_iterations": 1,
            "gauss_newton_max_iterations": 1,
            "verbose": 0,
        },
        anti_degeneracy_config={
            "per_angle_mode": "individual",
            "hierarchical": {
                "enable": True,
                "max_outer_iterations": 1,
                "physical_max_iterations": 2,
                "per_angle_max_iterations": 2,
            },
        },
    )
    assert info["covariance_is_placeholder"] is True


# ----------------------------------------------------------------- B10 / B11
def _enable_generic_streaming(monkeypatch: pytest.MonkeyPatch, factory: Any) -> Any:
    from tests.optimization.test_hybrid_streaming_retry import _fake_hybrid_streaming_config
    from xpcsjax.optimization.nlsq.strategies import hybrid_streaming as hs

    monkeypatch.setattr(hs, "HYBRID_STREAMING_AVAILABLE", True)
    monkeypatch.setattr(hs, "AdaptiveHybridStreamingOptimizer", factory)
    monkeypatch.setattr(hs, "HybridStreamingConfig", _fake_hybrid_streaming_config)
    return hs


# optstrat-B10/B11 reclassified DESIGN by the lead: the fallback-chain STREAMING tier has never
# run (always AttributeError -> CHUNKED); repairing it activates an unexercised solver and changes
# large-data fit results. Maintainer decision: repair or remove the tier. Tests removed.


# -------------------------------------------------------------------------- B15
def test_optstrat_b15(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fixed parameter must not make the sequential path drop a valid x_scale."""
    from xpcsjax.optimization.nlsq.strategies import sequential as seq

    seen: dict[str, Any] = {}

    def _spy(subset: Any, residual_func: Any, p0: Any, bounds: Any, **kw: Any) -> dict:
        seen.update(kw)
        return {
            "parameters": np.asarray(p0),
            "covariance": np.eye(len(p0)),
            "cost": 0.0,
            "success": True,
            "n_iterations": 1,
            "message": "",
            "n_points": subset.n_points,
            "phi_angle": subset.phi_angle,
            "jac_initial_norms": None,
            "jac_final_norms": None,
        }

    monkeypatch.setattr(seq, "optimize_single_angle", _spy)
    n = 12
    seq.optimize_per_angle_sequential(
        phi=np.zeros(n),
        t1=np.arange(float(n)),
        t2=np.arange(float(n)),
        g2_exp=np.ones(n),
        residual_func=lambda p, phi, t1, t2, g: g - p[0],
        initial_params=np.array([1.0, 1.0]),
        bounds=(np.array([0.0, 1.0]), np.array([3.0, 1.0])),  # 2nd parameter fixed
        x_scale=np.array([2.0, 7.0]),
    )
    assert "x_scale" in seen
    np.testing.assert_allclose(seen["x_scale"], [2.0])


