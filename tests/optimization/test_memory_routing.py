"""Direct unit tests for memory-aware NLSQ strategy routing.

Localizes router regressions ahead of the Phase 5 characterization gate."""

import pytest

from xpcsjax.optimization.nlsq.memory import select_nlsq_strategy


def _strategy_name(s) -> str:
    """Normalize the returned value (enum, string, dataclass) to upper-case name."""
    if hasattr(s, "strategy"):
        return getattr(s.strategy, "name", str(s.strategy)).upper()
    return getattr(s, "name", str(s)).upper()


def test_small_data_routes_to_standard():
    """Small datasets fit in memory — STANDARD strategy."""
    decision = select_nlsq_strategy(n_points=10_000, n_params=3)
    name = _strategy_name(decision)
    assert "STANDARD" in name, f"expected STANDARD, got {name}"


def test_large_data_with_tight_threshold_escalates():
    """When peak Jacobian memory exceeds the adaptive threshold, the router escalates.

    Uses memory_fraction=0.1 (minimum after clamping) and pins ``concurrency=1``
    so the assertion is deterministic regardless of the run context (serial vs
    pytest-xdist, which would otherwise shrink the budget further). Sizes are
    derived from the *actual* threshold the router computes (available-basis), so
    the test stays machine-portable across RAM sizes."""
    from xpcsjax.optimization.nlsq.memory import get_adaptive_memory_threshold

    threshold_gb, _ = get_adaptive_memory_threshold(0.1, concurrency=1)
    # Peak Jacobian = n_points * 14 * 8 * 6.5 bytes. Want peak > 2× threshold.
    n_points = int(2 * threshold_gb * (1024**3) / (14 * 8 * 6.5))

    decision = select_nlsq_strategy(
        n_points=n_points, n_params=14, memory_fraction=0.1, concurrency=1
    )
    name = _strategy_name(decision)
    assert any(token in name for token in ("OUT_OF_CORE", "CHUNK", "STREAM", "HYBRID")), (
        f"expected escalation beyond STANDARD on n_points={n_points} with mem_fraction=0.1, "
        f"got {name}"
    )


def test_memory_fraction_clamped_to_valid_range():
    """memory_fraction below 0.1 or above 0.9 is clamped (with a warning)."""
    with pytest.warns(UserWarning, match="clamped"):
        decision = select_nlsq_strategy(n_points=2_000_000, n_params=3, memory_fraction=0.001)
    name = _strategy_name(decision)
    assert name in {"STANDARD", "OUT_OF_CORE", "HYBRID_STREAMING"}


def test_n_params_zero_yields_zero_peak_and_standard():
    """Audit finding #17: n_params<=0 forces peak_memory_gb to 0.0 (can't estimate
    a Jacobian) and, with a small index, routes to STANDARD."""
    decision = select_nlsq_strategy(n_points=10_000, n_params=0)
    assert decision.peak_memory_gb == 0.0
    assert _strategy_name(decision) == "STANDARD"


def test_out_of_core_isolated_from_hybrid_streaming():
    """Audit finding #17: exercise the OUT_OF_CORE branch specifically — peak
    Jacobian exceeds the threshold while the int64 index array stays under it (so
    HYBRID_STREAMING, which checks the index first, does not fire)."""
    from xpcsjax.optimization.nlsq.memory import get_adaptive_memory_threshold

    # Pin concurrency=1 and derive sizes from the actual (available-basis)
    # threshold so the index<threshold<peak bracketing holds in any run context.
    threshold_gb, _ = get_adaptive_memory_threshold(0.1, concurrency=1)
    threshold_points = threshold_gb * (1024**3) / 8.0
    # index ~ 0.5 * threshold (< threshold); peak ~ n_points * 50 * 8 * 6.5 >> threshold.
    n_points = int(threshold_points / 2)

    decision = select_nlsq_strategy(
        n_points=n_points, n_params=50, memory_fraction=0.1, concurrency=1
    )
    assert _strategy_name(decision) == "OUT_OF_CORE", (
        f"expected OUT_OF_CORE for index<threshold<peak, got {_strategy_name(decision)}"
    )


def test_router_executes_without_exception_for_typical_inputs():
    """Smoke check: the router accepts XPCS-typical sizes without crashing."""
    for n_points, n_params in [
        (50_000, 3),
        (5_000_000, 7),
        (50_000_000, 14),
    ]:
        decision = select_nlsq_strategy(n_points=n_points, n_params=n_params)
        assert _strategy_name(decision) in {"STANDARD", "OUT_OF_CORE", "HYBRID_STREAMING"}
