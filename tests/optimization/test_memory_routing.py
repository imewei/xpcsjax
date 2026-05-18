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

    Uses memory_fraction=0.1 (minimum after clamping). On a 62.5 GB box that
    yields a 6.25 GB threshold; 100M points × 14 params × 8 bytes = ~11 GB peak
    Jacobian, which exceeds the threshold and triggers OUT_OF_CORE or beyond.
    On larger hosts (>100 GB RAM) we bump up n_points so the assertion stays
    machine-portable."""
    import psutil

    total_gb = psutil.virtual_memory().total / 1e9
    threshold_gb = 0.1 * total_gb  # mirrors the router's clamped fraction floor
    # Peak Jacobian = n_points * 14 * 8 bytes. Want peak > 2× threshold.
    n_points = max(100_000_000, int(2 * threshold_gb * 1e9 / (14 * 8)))

    decision = select_nlsq_strategy(
        n_points=n_points, n_params=14, memory_fraction=0.1
    )
    name = _strategy_name(decision)
    assert any(token in name for token in ("OUT_OF_CORE", "CHUNK", "STREAM", "HYBRID")), (
        f"expected escalation beyond STANDARD on n_points={n_points} with mem_fraction=0.1, "
        f"got {name}"
    )


def test_memory_fraction_clamped_to_valid_range():
    """memory_fraction below 0.1 or above 0.9 is clamped (with a warning)."""
    with pytest.warns(UserWarning, match="clamped"):
        decision = select_nlsq_strategy(
            n_points=2_000_000, n_params=3, memory_fraction=0.001
        )
    name = _strategy_name(decision)
    assert name in {"STANDARD", "OUT_OF_CORE", "HYBRID_STREAMING"}


def test_router_executes_without_exception_for_typical_inputs():
    """Smoke check: the router accepts XPCS-typical sizes without crashing."""
    for n_points, n_params in [
        (50_000, 3),
        (5_000_000, 7),
        (50_000_000, 14),
    ]:
        decision = select_nlsq_strategy(n_points=n_points, n_params=n_params)
        assert _strategy_name(decision) in {"STANDARD", "OUT_OF_CORE", "HYBRID_STREAMING"}
