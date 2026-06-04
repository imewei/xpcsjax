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
from xpcsjax.optimization.nlsq.heterodyne_core import _PER_ANGLE_CMAES_SEED, _fit_cmaes

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


@pytest.mark.skipif(
    not getattr(heterodyne_core, "HAS_CMAES", False),
    reason="CMA-ES backend (evosax) not installed; determinism not testable",
)
def test_per_angle_cmaes_is_bit_reproducible() -> None:
    """Two independent per-angle CMA-ES fits return identical parameters.

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
