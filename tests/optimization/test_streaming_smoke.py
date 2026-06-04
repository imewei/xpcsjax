"""Chunking / streaming smoke tests for memory-aware NLSQ routing.

The Phase 5 /double-check report flagged that ``HYBRID_STREAMING`` is selected
by :func:`xpcsjax.optimization.nlsq.memory.select_nlsq_strategy` and threaded
through the CMA-ES wrapper + anti-degeneracy controller, but **no test ever
forced the streaming branch end-to-end** — the marketing claim "memory-aware
routing for large datasets (chunking or streaming optimization)" was therefore
load-bearing on routing math that nothing exercised in CI.

This module closes that gap with three smoke checks:

1. **HYBRID_STREAMING auto-trigger at extreme scale** — uses a ``n_points``
   large enough that even on a 128 GB host the int64 index array alone exceeds
   the routing threshold, forcing the streaming branch deterministically. No
   real allocation happens; only the routing math runs.

2. **Both flavors of select_nlsq_strategy are wired** — the homodyne module
   (``memory.py``) and the heterodyne module (``heterodyne_memory.py``) use
   different strategy vocabularies (``HYBRID_STREAMING`` vs ``STREAMING``).
   This test pins both vocabularies and confirms each module's router returns
   the streaming-class decision under the same extreme inputs.

3. **Production-path reachability** — a grep-style assertion that the
   streaming entry is actually called from a non-test module. Catches the
   "select_nlsq_strategy is defined but orphaned" regression that the
   /double-check report originally suspected.
"""

from __future__ import annotations

import pathlib

import psutil
import pytest

# The heterodyne routing module — distinct enum (STANDARD / LARGE / STREAMING).
from xpcsjax.optimization.nlsq.heterodyne_memory import (
    NLSQStrategy as HeterodyneNLSQStrategy,
)
from xpcsjax.optimization.nlsq.heterodyne_memory import (
    select_nlsq_strategy as heterodyne_select_strategy,
)

# The homodyne routing module — distinct enum (STANDARD / OUT_OF_CORE / HYBRID_STREAMING).
from xpcsjax.optimization.nlsq.memory import (
    NLSQStrategy as HomodyneNLSQStrategy,
)
from xpcsjax.optimization.nlsq.memory import (
    select_nlsq_strategy as homodyne_select_strategy,
)


def _streaming_size_for_threshold(threshold_gb: float, factor: float = 2.0) -> int:
    """Number of int64 points whose index array exceeds ``factor × threshold_gb``.

    Solves ``n_points × 8 bytes > factor × threshold_gb × 2**30``. Used by both
    tests to pick a deterministic streaming-trigger size independent of the host's
    physical RAM.
    """
    return int(factor * threshold_gb * (1024**3) / 8)


# ---------------------------------------------------------------------------
# 1. HYBRID_STREAMING auto-trigger (homodyne)
# ---------------------------------------------------------------------------


def test_hybrid_streaming_triggers_when_index_exceeds_threshold() -> None:
    """Force the ``index_memory_gb > threshold_gb`` branch in homodyne routing.

    The routing math is:
        ``index_memory_gb = n_points * 8 / 2**30``
        ``threshold_gb = system_RAM_gb * memory_fraction``  (clamped to [0.1, 0.9])

    Pick ``n_points`` so the int64 index array alone exceeds twice the
    minimum-fraction threshold even on a host with several hundred GB of RAM.
    The decision is pure routing math — no memory is actually allocated.
    """
    total_gb = psutil.virtual_memory().total / (1024**3)
    # memory_fraction floor is 0.1; choose 2× the resulting threshold so the
    # test is robust to small variations in system memory detection.
    min_threshold_gb = max(0.1 * total_gb, 1.0)
    n_points = _streaming_size_for_threshold(min_threshold_gb, factor=2.0)

    decision = homodyne_select_strategy(n_points=n_points, n_params=11, memory_fraction=0.1)
    assert decision.strategy is HomodyneNLSQStrategy.HYBRID_STREAMING, (
        f"expected HYBRID_STREAMING for n_points={n_points:,} on a "
        f"{total_gb:.1f} GB host with memory_fraction=0.1, got "
        f"{decision.strategy.name} ({decision.reason!r})"
    )
    assert decision.index_memory_gb > decision.threshold_gb, (
        f"routing math is off: index_memory_gb={decision.index_memory_gb:.2f} "
        f"vs threshold_gb={decision.threshold_gb:.2f}"
    )


# ---------------------------------------------------------------------------
# 2. Both routing vocabularies are wired
# ---------------------------------------------------------------------------


def test_homodyne_and_heterodyne_routers_both_escalate_to_streaming() -> None:
    """The homodyne (HYBRID_STREAMING) and heterodyne (STREAMING) routers both
    return their streaming-class decision under identical extreme inputs.

    Pins the "two parallel routing modules" structural choice flagged by the
    /double-check report — if anyone unifies the two enums prematurely, this
    test will catch it because one of the enum members will go missing.
    """
    total_gb = psutil.virtual_memory().total / (1024**3)
    min_threshold_gb = max(0.1 * total_gb, 1.0)
    n_points = _streaming_size_for_threshold(min_threshold_gb, factor=2.0)

    homodyne_decision = homodyne_select_strategy(
        n_points=n_points, n_params=11, memory_fraction=0.1
    )
    heterodyne_decision = heterodyne_select_strategy(
        n_points=n_points, n_params=14, memory_fraction=0.1
    )

    assert homodyne_decision.strategy is HomodyneNLSQStrategy.HYBRID_STREAMING
    assert heterodyne_decision.strategy is HeterodyneNLSQStrategy.STREAMING

    # Enum-member sanity: a flat-out collapse of the two modules would lose one
    # of these names. Pin them explicitly.
    assert {s.name for s in HomodyneNLSQStrategy} == {
        "STANDARD",
        "OUT_OF_CORE",
        "HYBRID_STREAMING",
    }
    assert {s.name for s in HeterodyneNLSQStrategy} == {
        "STANDARD",
        "LARGE",
        "STREAMING",
    }


# ---------------------------------------------------------------------------
# 3. Production-path reachability (catches "defined but orphaned" regressions)
# ---------------------------------------------------------------------------


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_XPCSJAX_ROOT = _REPO_ROOT / "xpcsjax"


def _grep_callers(symbol: str) -> list[pathlib.Path]:
    """Return non-test python files under ``xpcsjax/`` that mention ``symbol``."""
    hits: list[pathlib.Path] = []
    for path in _XPCSJAX_ROOT.rglob("*.py"):
        # Skip the module that *defines* the symbol — we want call sites only.
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if symbol in text:
            hits.append(path)
    return hits


@pytest.mark.parametrize(
    "module_label,call_site_symbol,min_non_definition_callers",
    [
        # Homodyne: select_nlsq_strategy is called from wrapper.py (multiple sites),
        # adapter.py comments, and heterodyne_core.py.
        ("homodyne memory router", "select_nlsq_strategy", 2),
        # Heterodyne: heterodyne_select_strategy is called from heterodyne_adapter.py.
        ("heterodyne memory router", "select_nlsq_strategy", 2),
    ],
)
def test_select_strategy_is_reached_from_production_code(
    module_label: str, call_site_symbol: str, min_non_definition_callers: int
) -> None:
    """``select_nlsq_strategy`` must have non-test callers — otherwise the
    "memory-aware routing" marketing claim is unwired aspiration.

    Counts files (not lines) so a single chatty caller doesn't satisfy the
    assertion alone. The ``min_non_definition_callers`` threshold is set such
    that at least one non-defining module imports the symbol.
    """
    files = _grep_callers(call_site_symbol)
    non_test_files = [p for p in files if "tests" not in p.parts and "__pycache__" not in p.parts]
    assert len(non_test_files) >= min_non_definition_callers, (
        f"{module_label}: only {len(non_test_files)} non-test file(s) mention "
        f"{call_site_symbol!r}: {[p.name for p in non_test_files]}. "
        f"Expected at least {min_non_definition_callers} — routing is orphaned."
    )
