"""Tests for the JAX-free failure->message mapping."""

from xpcsjax.gui.error_presenter import present_failure


def test_oom_message_is_friendly():
    title, friendly, details = present_failure(
        "Worker exited abnormally (exit_code=-9, signal=9) — likely out of memory."
    )
    assert "memory" in friendly.lower()
    assert "exit_code" in details  # raw text retained for the details pane


def test_traceback_is_summarized_with_details_retained():
    tb = "Traceback (most recent call last):\n  ...\nValueError: bad config: missing analysis_mode"
    title, friendly, details = present_failure(tb)
    assert "ValueError: bad config: missing analysis_mode" in friendly  # last line surfaced
    assert details == tb  # full traceback kept for the expander


def test_plain_message_passthrough():
    title, friendly, details = present_failure("Fit did not converge")
    assert friendly == "Fit did not converge"
