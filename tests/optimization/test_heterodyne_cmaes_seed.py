"""Regression tests for per-angle CMA-ES reproducibility + config plumbing.

Guards the two defects fixed alongside this file:

1. **Missing seed (P2, reproducibility).** ``_fit_cmaes`` built its
   ``CMAESWrapperConfig`` without ``seed=``, so the per-angle CMA-ES escape
   left ``seed=None`` → non-reproducible run-to-run, unlike the seed-pinned
   joint escapes (``_JOINT_CMAES_SEED``). The fix pins
   ``_PER_ANGLE_CMAES_SEED + angle_idx`` — reproducible *and* decorrelated
   across angles (each angle gets a distinct-but-fixed seed).

2. **Dropped ``cmaes_sigma0`` (P3, silent config loss).** The heterodyne
   ``NLSQConfig.cmaes_sigma0`` (CMA-ES *initial step size*) was never mapped to
   ``CMAESWrapperConfig.sigma``, so the configured value was silently discarded
   and CMA-ES used the wrapper default (0.5). NOTE: this ``sigma`` (initial
   step) is unrelated to the ``sigma=`` *argument* of ``fit_with_cmaes``, which
   is the per-point measurement uncertainty and was always passed correctly.

The first three tests use a **spy** on ``fit_with_cmaes`` that captures the
``CMAESWrapperConfig`` and short-circuits — so they assert the exact contract
without requiring the evosax backend. The static guard parses the source so a
future fourth ``CMAESWrapperConfig(...)`` call site that forgets ``seed=`` fails
loudly. The determinism test is the gold-standard behavioral check and is gated
on evosax availability.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from xpcsjax.optimization.nlsq import heterodyne_core
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import (
    _PER_ANGLE_CMAES_SEED,
    _cmaes_joint_candidate,
    _fit_cmaes,
    _fit_joint_cmaes_multi_phi,
)

from ._heterodyne_fixtures import make_synthetic_two_component


class _CaptureAndStopError(Exception):
    """Raised by the spy after recording the config, to short-circuit
    ``_fit_cmaes`` before Phase 3 (which we don't exercise here)."""


def _install_capturing_spy(monkeypatch) -> dict:
    """Replace ``heterodyne_core.fit_with_cmaes`` with a spy that records the
    ``CMAESWrapperConfig`` it was handed, then raises ``_CaptureAndStopError``.

    ``_fit_cmaes`` calls ``fit_with_cmaes`` entirely by keyword, so the spy can
    accept ``**kwargs`` and pull ``config`` out unambiguously.
    """
    captured: dict = {}

    def spy(**kwargs):
        captured["config"] = kwargs["config"]
        captured["sigma_arg"] = kwargs.get("sigma")
        raise _CaptureAndStopError

    monkeypatch.setattr(heterodyne_core, "fit_with_cmaes", spy)
    return captured


def test_per_angle_cmaes_seed_pinned_and_angle_offset(monkeypatch) -> None:
    """Each per-angle CMA-ES config pins ``_PER_ANGLE_CMAES_SEED + angle_idx``.

    Regression: the config previously omitted ``seed=`` entirely, leaving it at
    ``None`` (non-reproducible). The angle offset is what keeps the N searches
    decorrelated while still individually reproducible.
    """
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=8)
    config = NLSQConfig(enable_cmaes=True, cmaes_warmstart_auto_skip=False)
    captured = _install_capturing_spy(monkeypatch)

    for angle_idx in (0, 2):
        captured.clear()
        with pytest.raises(_CaptureAndStopError):
            _fit_cmaes(
                model,
                c2[angle_idx],
                float(phi[angle_idx]),
                config,
                weights=None,
                angle_idx=angle_idx,
            )
        seed = captured["config"].seed
        assert seed is not None, (
            "per-angle CMA-ES must pin a seed (reproducibility); got None. "
            "This is the exact regression: CMAESWrapperConfig.seed defaulting "
            "to None makes the global search non-reproducible run-to-run."
        )
        assert seed == _PER_ANGLE_CMAES_SEED + angle_idx, (
            f"seed must be _PER_ANGLE_CMAES_SEED ({_PER_ANGLE_CMAES_SEED}) + "
            f"angle_idx ({angle_idx}); got {seed}"
        )


def test_per_angle_cmaes_sigma0_is_honored(monkeypatch) -> None:
    """The configured ``cmaes_sigma0`` reaches ``CMAESWrapperConfig.sigma``.

    Use a non-default value (0.17) so a coincidental match with the wrapper
    default (0.5) can't make a broken mapping pass.
    """
    model, c2, phi = make_synthetic_two_component(n_phi=1, n_t=8)
    config = NLSQConfig(enable_cmaes=True, cmaes_sigma0=0.17, cmaes_warmstart_auto_skip=False)
    captured = _install_capturing_spy(monkeypatch)

    with pytest.raises(_CaptureAndStopError):
        _fit_cmaes(model, c2[0], float(phi[0]), config, weights=None, angle_idx=0)

    assert captured["config"].sigma == pytest.approx(0.17), (
        "cmaes_sigma0 (initial step size) must be threaded into "
        f"CMAESWrapperConfig.sigma; got {captured['config'].sigma}. "
        "Regression: the field was silently dropped and the wrapper used 0.5."
    )


def test_all_cmaes_config_sites_pin_seed() -> None:
    """Every ``CMAESWrapperConfig(...)`` literal in heterodyne_core passes ``seed=``.

    Prevention guard for the root cause: the bug existed because three call
    sites were edited independently and one (the per-angle path) was missed.
    A new call site that forgets ``seed=`` fails here rather than silently
    shipping a non-reproducible escape.
    """
    source = inspect.getsource(heterodyne_core)
    tree = ast.parse(source)

    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        if name != "CMAESWrapperConfig":
            continue
        kwarg_names = {kw.arg for kw in node.keywords}
        if "seed" not in kwarg_names:
            offenders.append(node.lineno)

    assert not offenders, (
        "every CMAESWrapperConfig(...) in heterodyne_core.py must pin seed= for "
        f"reproducibility; sites missing seed= at lines: {offenders}"
    )


def test_all_cmaes_config_sites_pin_sigma() -> None:
    """Every ``CMAESWrapperConfig(...)`` literal in heterodyne_core passes ``sigma=``.

    Prevention guard for the same root cause that dropped ``cmaes_sigma0`` on the
    joint escapes: the per-angle site threaded ``sigma=`` while the two joint
    sites silently used the wrapper default (0.5). A new call site that forgets
    ``sigma=`` fails here rather than shipping the wrong initial step size.
    """
    source = inspect.getsource(heterodyne_core)
    tree = ast.parse(source)

    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        if name != "CMAESWrapperConfig":
            continue
        kwarg_names = {kw.arg for kw in node.keywords}
        if "sigma" not in kwarg_names:
            offenders.append(node.lineno)

    assert not offenders, (
        "every CMAESWrapperConfig(...) in heterodyne_core.py must pin sigma= "
        "(cmaes_sigma0 initial step size); sites missing sigma= at lines: "
        f"{offenders}"
    )


@pytest.mark.skipif(
    not getattr(heterodyne_core, "HAS_CMAES", False),
    reason="CMA-ES backend (evosax) not installed; determinism not testable",
)
def test_per_angle_cmaes_is_bit_reproducible() -> None:
    """Two separate per-angle CMA-ES fits return identical parameters.

    The behavioral consequence of the seed fix. A fresh model is built per run
    because ``model.scaling`` is mutated by every fit (see heterodyne_core's
    "same seed → same result" caveat) — reusing one model would let run 2 start
    from run 1's mutated state and defeat the comparison.
    """
    config = NLSQConfig(
        enable_cmaes=True,
        cmaes_warmstart_auto_skip=False,
        cmaes_max_iterations=15,
        cmaes_population_size=6,
        cmaes_restart_strategy="none",
        cmaes_max_restarts=0,
    )

    params = []
    for _ in range(2):
        # Fresh model each run; identical data (fixture uses a fixed RNG seed).
        model, c2, phi = make_synthetic_two_component(n_phi=1, n_t=10)
        result = _fit_cmaes(model, c2[0], float(phi[0]), config, weights=None, angle_idx=0)
        params.append(np.asarray(result.parameters, dtype=np.float64))

    np.testing.assert_array_equal(
        params[0],
        params[1],
        err_msg=(
            "per-angle CMA-ES must be bit-reproducible with the pinned seed; "
            "differing parameters mean the seed is not being forwarded "
            "deterministically into the evosax backend."
        ),
    )


def test_joint_cmaes_candidate_sigma0_is_honored(monkeypatch) -> None:
    """``_cmaes_joint_candidate`` (the averaged/constant joint-escape hook)
    threads ``cmaes_sigma0`` into ``CMAESWrapperConfig.sigma``, mirroring
    ``test_per_angle_cmaes_sigma0_is_honored`` but at the joint-escape site.

    ``test_all_cmaes_config_sites_pin_sigma`` only proves every
    ``CMAESWrapperConfig(...)`` literal passes SOME ``sigma=`` kwarg (a static
    AST presence check); it cannot catch a value-level regression such as
    ``sigma=0.5`` (a hardcoded default) or a swapped field. This test pins the
    actual float.
    """
    config = NLSQConfig(cmaes_sigma0=0.17)
    x_warm = np.array([1.0, 2.0, 3.0])
    lb = np.zeros(3)
    ub = np.full(3, 10.0)
    captured = _install_capturing_spy(monkeypatch)

    with pytest.raises(_CaptureAndStopError):
        _cmaes_joint_candidate(lambda x: np.asarray(x, dtype=np.float64), x_warm, lb, ub, config)

    assert captured["config"].sigma == pytest.approx(0.17), (
        "cmaes_sigma0 must be threaded into CMAESWrapperConfig.sigma at the "
        f"averaged/constant joint-escape site; got {captured['config'].sigma}. "
        "Regression: this site could silently fall back to the wrapper's 0.5 "
        "default while test_all_cmaes_config_sites_pin_sigma stays green "
        "(it only checks the kwarg is present, not its value)."
    )


def test_joint_multi_phi_cmaes_sigma0_is_honored(monkeypatch) -> None:
    """``_fit_joint_cmaes_multi_phi`` (the individual-mode joint-escape entry)
    threads ``cmaes_sigma0`` into ``CMAESWrapperConfig.sigma``.

    Unlike ``_cmaes_joint_candidate``, this function wraps its body in a
    best-effort ``except Exception: fall back to the plain joint fit`` — so a
    raise-and-capture spy would be silently swallowed and never surface the
    regression. Instead the spy returns a non-raising, unsuccessful result
    (``success=False``): ``_cmaes_keep_best_over_seeds`` short-circuits on
    ``success`` before touching ``.parameters``, so the escape completes
    normally (keeping the warm-start) while still letting us inspect the
    captured config.
    """
    from types import SimpleNamespace

    model, c2, phi = make_synthetic_two_component(n_phi=2, n_t=8)
    config = NLSQConfig(enable_cmaes=True, cmaes_sigma0=0.17, cmaes_warmstart_auto_skip=False)

    captured: dict = {}

    def spy(**kwargs):
        captured["config"] = kwargs["config"]
        return SimpleNamespace(success=False, parameters=None)

    monkeypatch.setattr(heterodyne_core, "fit_with_cmaes", spy)

    result = _fit_joint_cmaes_multi_phi(model, c2, phi, config, weights=None)

    assert result is not None and result.parameters is not None, (
        "the escape must fall back to the warm-start (not raise/crash) when "
        "the spy reports an unsuccessful CMA-ES draw"
    )
    assert "config" in captured, "fit_with_cmaes must have been invoked"
    assert captured["config"].sigma == pytest.approx(0.17), (
        "cmaes_sigma0 must be threaded into CMAESWrapperConfig.sigma at the "
        f"individual-mode joint-escape site; got {captured['config'].sigma}. "
        "Regression: this site could silently fall back to the wrapper's 0.5 "
        "default while test_all_cmaes_config_sites_pin_sigma stays green "
        "(it only checks the kwarg is present, not its value)."
    )


def test_fit_cmaes_dof_clamp_prevents_negative_reduced_chi_squared(monkeypatch) -> None:
    """``_fit_cmaes``'s post-fit χ² correction clamps ``n_dof_valid`` to at
    least 1 (``max(n_valid - n_params, 1)``) so a tiny matrix with more
    varying parameters than valid (off-diagonal, non-boundary) data points
    cannot drive the degrees-of-freedom negative.

    With ``n_phi=1, n_t=5``: ``n_valid = (5-1)*(5-2) = 12`` and the
    two-component per-angle model has 14 varying parameters, so
    ``n_valid - n_params = -2`` -- exactly the case the clamp guards. Without
    it, ``reduced_chi_squared`` would flip sign (negative), which is
    unphysical and would silently corrupt downstream quality classification.

    CMA-ES itself is stubbed to a non-raising, unsuccessful result (mirroring
    ``test_joint_multi_phi_cmaes_sigma0_is_honored``'s technique) purely to
    keep this test fast and evosax-independent; the DOF clamp under test
    runs in the shared post-Phase-3 block regardless of which optimizer won.
    """
    from types import SimpleNamespace

    model, c2, phi = make_synthetic_two_component(n_phi=1, n_t=5)
    n_params = len(model.param_manager.varying_names)
    n_matrix = c2.shape[1]
    n_valid = (n_matrix - 1) * (n_matrix - 2)
    assert n_valid - n_params < 0, (
        "fixture assumption broken: this test requires n_valid < n_params so "
        f"the clamp is exercised; got n_valid={n_valid}, n_params={n_params}. "
        "Adjust n_t if the model's varying-parameter count changes."
    )

    def spy(**kwargs):
        return SimpleNamespace(success=False, parameters=None, covariance=None, diagnostics={})

    monkeypatch.setattr(heterodyne_core, "fit_with_cmaes", spy)

    config = NLSQConfig(enable_cmaes=True, cmaes_warmstart_auto_skip=False)
    result = _fit_cmaes(model, c2[0], float(phi[0]), config, weights=None, angle_idx=0)

    assert result.success, "warm-start NLSQ fit must succeed for this DOF check to be meaningful"
    assert result.reduced_chi_squared is not None
    assert result.reduced_chi_squared > 0, (
        "reduced_chi_squared must stay positive even when n_valid < n_params; "
        f"got {result.reduced_chi_squared}. Regression: an unclamped "
        "n_dof_valid = n_valid - n_params (negative here) flips the sign of "
        "ssr / (sigma2_noise * n_dof_valid)."
    )
    assert np.isfinite(result.reduced_chi_squared)
