"""Regression: ``get_pressure_trend`` must not iterate ``_pressure_history`` bare.

``_update_stats`` runs on the monitoring daemon and appends to the deque every
``monitoring_interval`` (default 1s), while any thread calling
``get_memory_stats`` -> ``get_pressure_trend`` used to iterate that same deque
directly in a list comprehension, raising ``RuntimeError: deque mutated during
iteration``.

Against the unfixed code this test is probabilistic (a writer thread racing a
reader for a bounded number of iterations); against the fixed code it is
deterministically green. It is bounded to a few hundred iterations rather than
a wall-clock stress loop so it stays fast enough for the default suite.
"""

from __future__ import annotations

import sys
import threading

from xpcsjax.data.memory_manager import MemoryPressureMonitor


def test_get_pressure_trend_survives_concurrent_appends():
    monitor = MemoryPressureMonitor(warning_threshold=0.8, critical_threshold=0.9)
    # Stub the psutil poll so the writer loop is tight (and the test hermetic);
    # the deque append under test is what matters.
    monitor.stats.update_system_stats = lambda: None  # type: ignore[method-assign]

    # Prime past the <10 sample early-return so the trend path really iterates.
    monitor.stats.memory_pressure = 0.5
    for _ in range(20):
        monitor._update_stats()

    stop = threading.Event()
    writer_errors: list[BaseException] = []

    def writer() -> None:
        try:
            while not stop.is_set():
                monitor._update_stats()
        except BaseException as exc:  # pragma: no cover - failure path
            writer_errors.append(exc)

    thread = threading.Thread(target=writer, daemon=True)
    # A short switch interval forces thread hand-offs mid-comprehension, which is
    # what makes the unfixed code fail within a few hundred iterations instead of
    # needing a multi-second stress loop.
    original_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    thread.start()
    try:
        for _ in range(500):
            monitor.get_pressure_trend()
    finally:
        stop.set()
        thread.join(timeout=5.0)
        sys.setswitchinterval(original_interval)

    assert not writer_errors, f"writer thread raised: {writer_errors[0]!r}"
