"""Multi-seed keep-best for the heterodyne joint CMA-ES escape.

RCA (C044 two_component, 2026-06-16): the joint CMA-ES global search is NOT
run-to-run reproducible despite a pinned ``seed=42`` — XLA's float-reduction
order over the ~3M-point objective varies between runs, and the basin-fragile
non-convex search amplifies that into different basins (good fit SSR 5546,
beta=-0.41 vs degenerate flat-fan SSR 8737, beta=-0.03). The fix runs the escape
over N seeds and keeps the lowest DATA-ONLY SSR, raising the probability of
landing the good basin while staying strictly keep-better (a worse draw can
never be selected over a better one).

These are PURE unit tests of the selection helper ``_cmaes_keep_best_over_seeds``
— no heavy CMA-ES / JAX machinery. ``run_one_seed`` is a stub returning fake
results whose ``parameters[0]`` encodes the SSR the (stubbed) ``data_ssr`` reads
back, so the selection logic is exercised deterministically.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from xpcsjax.optimization.nlsq.heterodyne_core import _cmaes_keep_best_over_seeds


def _result(ssr: float, *, success: bool = True) -> SimpleNamespace:
    """Fake OptimizationResult: parameters[0] encodes the SSR data_ssr reads."""
    return SimpleNamespace(
        success=success,
        parameters=np.array([ssr], dtype=np.float64),
    )


def _data_ssr(params: np.ndarray) -> float:
    return float(np.asarray(params, dtype=np.float64)[0])


def test_keeps_lowest_ssr_across_seeds() -> None:
    # Seeds yield SSRs [8737 (bad), 5546 (good), 9000] — the good basin must win.
    ssr_by_seed = {42: 8737.0, 43: 5546.0, 44: 9000.0}
    best, best_ssr, best_seed = _cmaes_keep_best_over_seeds(
        run_one_seed=lambda s: _result(ssr_by_seed[s]),
        seeds=[42, 43, 44],
        data_ssr=_data_ssr,
    )
    assert best_seed == 43
    assert best_ssr == 5546.0
    assert best.parameters[0] == 5546.0


def test_single_seed_returns_that_run_verbatim() -> None:
    # n_seeds == 1 MUST be byte-identical to the pre-multiseed path: the single
    # run's result object is returned unchanged.
    sentinel = _result(6796.0)
    best, best_ssr, best_seed = _cmaes_keep_best_over_seeds(
        run_one_seed=lambda s: sentinel,
        seeds=[42],
        data_ssr=_data_ssr,
    )
    assert best is sentinel
    assert best_ssr == 6796.0
    assert best_seed == 42


def test_failed_runs_are_deprioritized_but_never_lose_a_result() -> None:
    # A failed run scores +inf SSR; a later successful run must be selected.
    def run(seed: int) -> SimpleNamespace:
        if seed == 42:
            return _result(0.0, success=False)  # failed: params present but unconverged
        return _result(7000.0)

    best, best_ssr, best_seed = _cmaes_keep_best_over_seeds(
        run_one_seed=run,
        seeds=[42, 43],
        data_ssr=_data_ssr,
    )
    assert best_seed == 43
    assert best_ssr == 7000.0


def test_all_failed_returns_first_with_inf_ssr() -> None:
    # When every seed fails, return the FIRST run (so the caller's keep-better
    # vs the warm-start still has a vector to fall back from) tagged ssr=inf.
    first = _result(0.0, success=False)
    best, best_ssr, best_seed = _cmaes_keep_best_over_seeds(
        run_one_seed=lambda s: first if s == 42 else _result(0.0, success=False),
        seeds=[42, 43],
        data_ssr=_data_ssr,
    )
    assert best is first
    assert best_seed == 42
    assert best_ssr == float("inf")


def test_ties_keep_the_earlier_seed() -> None:
    # Deterministic tie-break: equal SSR keeps the earlier seed, so n_seeds=1 and
    # the first seed of n_seeds>1 select identically.
    best, best_ssr, best_seed = _cmaes_keep_best_over_seeds(
        run_one_seed=lambda s: _result(5000.0),
        seeds=[42, 43, 44],
        data_ssr=_data_ssr,
    )
    assert best_seed == 42
    assert best_ssr == 5000.0


# ---------------------------------------------------------------------------
# Integration: the joint CMA-ES escape runs cmaes_n_seeds DISTINCT draws.
# ---------------------------------------------------------------------------
def test_joint_escape_runs_n_seeds_distinct_draws(monkeypatch) -> None:
    """``cmaes_n_seeds=3`` ⇒ ``fit_with_cmaes`` is invoked 3× with seeds 42/43/44.

    Each draw is stubbed as a failed CMA-ES (so the loop completes fast and the
    keep-better falls back to the warm-start), letting us assert ONLY the seed
    schedule — that the escape actually draws ``[_JOINT_CMAES_SEED + i]`` rather
    than one fixed seed. ``cmaes_warmstart_auto_skip=False`` forces the global
    search to run on the (good) synthetic warm-start.
    """
    import xpcsjax.optimization.nlsq.heterodyne_core as hc
    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig

    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=12)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "individual",
            "enable_cmaes": True,
            "cmaes_warmstart_auto_skip": False,
            "cmaes_n_seeds": 3,
        }
    )

    seen_seeds: list[int] = []

    def _fake_cmaes(*, config, **_kwargs):  # noqa: ANN001 - test stub
        seen_seeds.append(int(config.seed))
        return SimpleNamespace(success=False, parameters=None)

    monkeypatch.setattr(hc, "fit_with_cmaes", _fake_cmaes)

    result = hc._fit_joint_cmaes_multi_phi(model, c2, phi, cfg, None)

    assert seen_seeds == [42, 43, 44]
    # All draws failed ⇒ the escape kept the warm-start, not a CMA-ES vector.
    assert result.nlsq_diagnostics.get("global_escape") == "cmaes_warmstart_kept"
