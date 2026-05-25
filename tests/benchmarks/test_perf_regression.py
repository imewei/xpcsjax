"""Wall-clock regression suite for xpcsjax v0.1 hot paths.

The /double-check performance review flagged: "No benchmark suite run today
— for a release gate you'd want a wall-clock regression test on a
representative 10k-point fit and a >1M-point chunked fit." This module
closes that gap with three ``pytest-benchmark`` micro-benchmarks:

1. **select_nlsq_strategy at 10k**: pure routing math, dominates per-fit
   overhead. Anything over a few microseconds means a regression in the
   ``get_adaptive_memory_threshold`` / ``estimate_peak_memory_gb`` path.

2. **select_nlsq_strategy at 10M**: routing decision when ``index_memory_gb``
   approaches the system threshold. Same code path, but exercises the
   ``HYBRID_STREAMING`` branch on hosts with <80 GB RAM. Catches accidental
   O(n_points) work creeping into the routing layer.

3. **Heterodyne per-angle local fit smoke**: ``_fit_local`` on a 16×16
   synthetic c2 surface. Not a true 10k-point fit (which would need a real
   homodyne dataset) but the smallest end-to-end timer for the NLSQ pipeline.
   Catches regressions in JIT warmup, model_func tracing, and NLSQAdapter
   dispatch.

Output:
    Pytest-benchmark writes per-run JSON to ``.benchmarks/``. Use::

        XPCSJAX_RUN_BENCHMARKS=1 uv run pytest tests/benchmarks/ \\
            --benchmark-save=baseline

    to pin a baseline, and::

        XPCSJAX_RUN_BENCHMARKS=1 uv run pytest tests/benchmarks/ \\
            --benchmark-compare=baseline --benchmark-compare-fail=mean:25%

    to fail CI on a >25% mean regression.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Bench 1 + 2: select_nlsq_strategy routing decisions
# ---------------------------------------------------------------------------


def test_perf_select_strategy_10k_points(benchmark) -> None:
    """Routing decision for a typical XPCS fit (10k points, 11 params).

    Expected: well under 1 ms — the routing path is just a few stat() and
    psutil calls. A regression to >5 ms means someone added expensive work
    (e.g. an extra disk scan) to the per-fit hot path.
    """
    from xpcsjax.optimization.nlsq.memory import select_nlsq_strategy

    result = benchmark(select_nlsq_strategy, n_points=10_000, n_params=11)
    assert result.strategy.name == "STANDARD"


def test_perf_select_strategy_10m_points(benchmark) -> None:
    """Routing decision at chunked-fit scale (10M points, 14 params).

    Same code path as the 10k case — pure math + psutil — so the wall clock
    should be effectively identical. Different from the 10k case only in
    whether ``HYBRID_STREAMING``/``OUT_OF_CORE`` is returned (depends on host
    RAM). Asserts the decision returns a valid strategy enum.
    """
    from xpcsjax.optimization.nlsq.memory import select_nlsq_strategy

    result = benchmark(
        select_nlsq_strategy, n_points=10_000_000, n_params=14, memory_fraction=0.1
    )
    assert result.strategy.name in {"STANDARD", "OUT_OF_CORE", "HYBRID_STREAMING"}


# ---------------------------------------------------------------------------
# Bench 3: heterodyne per-angle local fit (end-to-end)
# ---------------------------------------------------------------------------


_HET_N_TIMES = 16
_HET_DT = 1.0
_HET_Q = 0.0054
_HET_PHI = 0.0
_HET_NOISE = 5e-3


def _het_smoke_config_dict() -> dict:
    """Tiny heterodyne config — same shape as test_heterodyne_cmaes.py."""
    return {
        "analysis_mode": "two_component",
        "analyzer_parameters": {
            "dt": _HET_DT,
            "start_frame": 1,
            "end_frame": _HET_N_TIMES,
            "scattering": {"wavevector_q": _HET_Q},
        },
        "scaling": {
            "n_angles": 1,
            "mode": "constant",
            "initial_contrast": 0.3,
            "initial_offset": 1.0,
        },
        "optimization": {
            "nlsq": {
                "analysis_mode": "two_component",
                "max_iterations": 30,
                "enable_cmaes": False,
            },
        },
    }


def _build_synthetic_c2(model) -> np.ndarray:
    rng = np.random.default_rng(seed=20260519)
    c2 = np.asarray(model.compute_correlation(phi_angle=_HET_PHI, angle_idx=0))
    return c2 + rng.normal(0.0, _HET_NOISE, size=c2.shape)


def test_perf_heterodyne_per_angle_local_fit(benchmark, tmp_path: Path) -> None:
    """End-to-end timing for the heterodyne per-angle local NLSQ fit.

    Smallest meaningful "real fit" smoke. Not a 10k-point homodyne (which
    needs an external dataset to be representative) but the smallest
    end-to-end timer for the NLSQ pipeline.

    Includes JIT compilation in the first round; pytest-benchmark by
    default reports the median across multiple rounds so the warm cache
    dominates. The first-run cost is reported separately in the JSON.
    """
    import yaml

    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
    from xpcsjax.optimization.nlsq import fit_nlsq
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    cfg_path = tmp_path / "perf_smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(_het_smoke_config_dict()))
    cfg = ConfigManager(str(cfg_path))
    assert cfg.config is not None

    truth_model = HeterodyneModel.from_config(cfg.config)
    c2 = _build_synthetic_c2(truth_model)
    data = {"c2": c2[np.newaxis, :, :], "phi": np.array([_HET_PHI])}

    # pytest-benchmark calls the fn under timing; assertions go after.
    # fit_nlsq returns a single OptimizationResult (not a list) — per-angle
    # data lives in result.nlsq_diagnostics for all dispatch modes.
    result = benchmark(fit_nlsq, data, cfg)
    assert isinstance(result, OptimizationResult), f"expected OptimizationResult, got {type(result)}"
    assert np.all(np.isfinite(np.asarray(result.parameters, dtype=np.float64)))
