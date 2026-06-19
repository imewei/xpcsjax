"""Map raw failure text to a user-facing (title, message, details) triple."""

from __future__ import annotations

_OOM_HINT = "likely out of memory"


def present_failure(message: str) -> tuple[str, str, str]:
    """Return ``(title, friendly_message, details)`` for a failure string.

    Parameters
    ----------
    message:
        The raw failure text emitted by the worker (traceback, worker-died
        message, or plain error string).

    Returns
    -------
    tuple[str, str, str]
        A ``(title, friendly_message, details)`` triple where:

        - *title* is a short dialog title.
        - *friendly_message* is shown prominently to the user (no raw traceback).
        - *details* is the full original text, placed behind a "Show details"
          collapsible pane.

    Notes
    -----
    Three cases are handled:

    - A worker "killed (likely out of memory)" message → a memory-pressure hint.
    - A Python traceback → its final non-empty line as the message, full
      traceback in details.
    - Anything else → passthrough (same text for both friendly and details).
    """
    text = (message or "").strip()
    if _OOM_HINT in text:
        return (
            "Fit worker stopped",
            "The fit was stopped before finishing — most likely it ran out of "
            "memory. Try a smaller angle subset or close other applications.",
            text,
        )
    if text.startswith("Traceback (most recent call last):"):
        last = next((ln for ln in reversed(text.splitlines()) if ln.strip()), text)
        return ("Fit failed", last.strip(), text)
    return ("Fit failed", text, text)
