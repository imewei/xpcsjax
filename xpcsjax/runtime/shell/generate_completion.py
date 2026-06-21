"""Generate xpcsjax/runtime/shell/completion.sh from the CLI parsers.

The completion script is a DERIVED artifact: this module is the single writer.
Run ``python -m xpcsjax.runtime.shell.generate_completion`` to regenerate, or
``make completion``. A parity test fails CI if the committed file drifts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

# Task 4 uses ONLY Hint. COMMAND_SPECS / CommandSpec are imported in Task 5 when
# the emitters that use them are added — importing them now would trip ruff F401
# (unused import) and fail this task's own lint step.
from xpcsjax.runtime.shell.completion_spec import Hint


@dataclass(frozen=True)
class Completion:
    """How to complete the value of one option. kind drives the emitted bash."""

    kind: str  # configfile | file | dir | threads | words | choices | none
    payload: str = ""


def classify_option(action: argparse.Action, hints: dict[str, Hint]) -> Completion | None:
    """Classify one option into a value-completion descriptor.

    Parameters
    ----------
    action : argparse.Action
        The argparse action representing a single CLI option.
    hints : dict[str, Hint]
        Per-command completion hints keyed by flag string (e.g. ``"--output"``).
        Values are either a string keyword or a tuple/list of literal words.

    Returns
    -------
    Completion | None
        ``None`` for a zero-argument flag (``nargs == 0``). A ``Completion``
        describing the value completion otherwise. Raises ``ValueError`` for an
        unknown hint string — by construction the final fallback is ``"none"``
        so this only fires on future unknown hint strings.
    """
    # Zero-argument actions are flags: store_true/store_false/store_const/count/
    # version/help all set nargs == 0. Classify on nargs, never an allow-list.
    if action.nargs == 0:
        return None

    # Per-command dynamic hints take precedence over type-based inference.
    flag = action.option_strings[0] if action.option_strings else ""
    hint = _lookup_hint(list(action.option_strings), hints)
    if hint is not None:
        if isinstance(hint, (tuple, list)):  # literal word list
            return Completion(kind="words", payload=" ".join(hint))
        if hint in {"configfile", "file", "dir", "threads"}:
            return Completion(kind=hint)
        raise ValueError(f"unknown hint {hint!r} for {flag}")

    if action.choices:
        return Completion(kind="choices", payload=" ".join(str(c) for c in action.choices))

    if action.type is Path:
        return Completion(kind="file")

    # Takes a value but no completion source: emit nothing.
    return Completion(kind="none")


def _lookup_hint(option_strings: list[str], hints: dict[str, Hint]) -> Hint | None:
    """Return the first matching hint for any of the option's flag strings.

    Parameters
    ----------
    option_strings : list[str]
        All flag strings for the action (e.g. ``["-o", "--output"]``).
    hints : dict[str, Hint]
        Hint mapping keyed by flag string.

    Returns
    -------
    Hint | None
        The first matching hint, or ``None`` if none match.
    """
    for s in option_strings:
        if s in hints:
            return hints[s]
    return None
