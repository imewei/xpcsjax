"""Regression: pressure-state hysteresis dead-zone must preserve prior state.

``_check_pressure_levels`` used to pre-initialize ``new_state = "normal"`` before
the critical/warning/recovery chain. In the hysteresis dead-zone between
``warning_threshold*0.8`` and ``warning_threshold`` NONE of the three branches
fire, yet the trailing ``self._last_pressure_state = new_state`` forced the state
to "normal" anyway — WITHOUT going through the recovery branch that resets the
latch flags and fires the recovery response. A subsequent genuine recovery then
saw a falsely-"normal" state and skipped the recovery callback entirely.

The fix initializes ``new_state = self._last_pressure_state`` so the dead-zone
preserves the previous state; only the true recovery branch transitions to
"normal".
"""

from xpcsjax.data.memory_manager import MemoryPressureMonitor


def test_deadzone_preserves_state_and_recovery_still_fires():
    """0.85 -> 0.75 (dead zone) -> 0.5: recovery must fire exactly once.

    With warning_threshold=0.8 the dead zone is [0.64, 0.8). 0.75 sits inside it.
    Under the old hardcoded ``new_state="normal"`` the 0.75 step corrupts the
    state to "normal", so the recovery branch at 0.5 sees a non-warning state and
    never fires. The fix keeps the state "warning" through the dead zone.
    """
    monitor = MemoryPressureMonitor(warning_threshold=0.8, critical_threshold=0.9)

    warning_calls = []
    recovery_calls = []
    monitor._trigger_warning_response = lambda: warning_calls.append(1)  # type: ignore[method-assign]
    monitor._trigger_recovery_response = lambda: recovery_calls.append(1)  # type: ignore[method-assign]

    # Step 1: cross the warning threshold -> warning fires, state latches.
    monitor.stats.memory_pressure = 0.85
    monitor._check_pressure_levels()
    assert warning_calls == [1]
    assert monitor._last_pressure_state == "warning"

    # Step 2: dead zone. No branch fires; state MUST remain "warning" (not reset
    # to "normal") so the latch flags stay consistent.
    monitor.stats.memory_pressure = 0.75
    monitor._check_pressure_levels()
    assert monitor._last_pressure_state == "warning", (
        "dead-zone must preserve the prior state, not force it to 'normal'"
    )
    assert warning_calls == [1], "no new warning should fire in the dead zone"

    # Step 3: genuine recovery below warning_threshold*0.8 -> recovery must fire.
    monitor.stats.memory_pressure = 0.5
    monitor._check_pressure_levels()
    assert recovery_calls == [1], (
        "recovery response must fire after a real drop; the dead-zone state "
        "corruption previously swallowed it"
    )
    assert monitor._last_pressure_state == "normal"
