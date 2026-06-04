"""Golden-snapshot load/init helper for the parity preservation suites.

Mechanism: if the golden ``.npz`` exists and we are NOT regenerating, load it and
return its arrays for the caller to assert against. If it is absent OR
``regen=True`` (driven by ``XPCSJAX_REGEN_GOLDEN=1``), build the payload, write it,
and return it (so the FIRST run writes the golden and a later run asserts).

On a normal run the golden must already exist, so the caller's assertions compare
live output against the committed snapshot — a behavior change fails.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def load_or_init_golden(
    path: Path,
    regen: bool,
    payload: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Return the golden arrays at ``path``.

    Args:
        path: Destination ``.npz`` path under ``tests/parity/_golden/``.
        regen: When True, always (re)write the golden from ``payload()``.
        payload: Zero-arg callable returning the dict of arrays to snapshot. Only
            invoked when (re)writing — never when asserting against an existing
            golden — so live values cannot leak into a load-and-assert run.

    Returns:
        Dict of numpy arrays loaded from disk (assert path) or freshly written
        (init/regen path).
    """
    if regen or not path.exists():
        data = payload()
        path.parent.mkdir(parents=True, exist_ok=True)
        # allow_pickle is required for the dtype=object string/key arrays.
        np.savez(path, **data)
        return data

    # allow_pickle=True is safe here: these goldens are committed in-repo and are
    # written ONLY by this same test suite (never loaded from an untrusted
    # source). It is needed solely to round-trip the dtype=object string/key
    # arrays (diagnostics keys, flag/status strings) snapshotted above.
    with np.load(path, allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}
