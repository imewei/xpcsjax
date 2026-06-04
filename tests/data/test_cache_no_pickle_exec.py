"""SEC-1 regression: the disk cache must never execute pickle payloads.

The quality-gate audit (owasp A04-1 / exploit-hunter F1) flagged
``MultiLevelCache._load_from_disk`` as a ``pickle.loads`` sink reachable from a
shared/multi-user cache directory — an arbitrary-code-execution vector that is
*unmitigated* on platforms without ``os.getuid`` (Windows). The fix replaces
pickle with ``np.save``/``np.load(allow_pickle=False)`` so the deserializer can
no longer construct arbitrary Python objects.

These tests pin the security property directly:

1. A crafted pickle planted in the cache (as an attacker who owns the file
   would) must be *refused*, and its ``__reduce__`` side effect must NOT run.
2. The legitimate payload type (a NumPy array) still round-trips.

If these fail, do not loosen them — the deserializer regressed back to a
code-execution sink.
"""

from __future__ import annotations

import pickle  # noqa: S403 — needed only to construct the hostile fixture
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Module-level sentinel toggled by the malicious payload's __reduce__ hook.
# If the loader ever unpickles the payload, this flips to True.
_EXEC_MARKER = {"ran": False}


def _trip_marker() -> str:
    _EXEC_MARKER["ran"] = True
    return "pwned"


class _Evil:
    """A payload whose unpickling executes ``_trip_marker`` via ``__reduce__``."""

    def __reduce__(self):  # noqa: D401 - pickle protocol hook
        return (_trip_marker, ())


@pytest.fixture
def cache_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from xpcsjax.data.performance_engine import MultiLevelCache

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    return MultiLevelCache(memory_cache_mb=1.0, ssd_cache_mb=1.0, hdd_cache_mb=1.0)


def _plant_pickle(engine: Any, key: str, payload: bytes) -> Path:
    """Write attacker-controlled bytes into the cache using the loader's framing.

    Mirrors an attacker who can write a file they own (0o600) into the cache
    directory — so the ownership/mode gates pass and the deserializer itself is
    what must reject the payload.
    """
    from xpcsjax.data import performance_engine as pe

    framed = pe.zstd.compress(payload) if pe.HAS_ZSTD else payload
    target = engine._ssd_cache_path / f"{key}.zstd"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(framed)
    target.chmod(0o600)
    return target


def test_planted_pickle_is_refused_and_not_executed(cache_engine: Any) -> None:
    """A pickle payload in the cache must not be deserialized or executed."""
    _EXEC_MARKER["ran"] = False
    payload = pickle.dumps(_Evil())
    target = _plant_pickle(cache_engine, "evil", payload)

    with pytest.raises((OSError, ValueError)):
        cache_engine._load_from_disk(target)

    assert _EXEC_MARKER["ran"] is False, (
        "the cache loader executed a pickle payload — RCE vector is open"
    )


def test_ndarray_round_trips_through_disk_cache(cache_engine: Any) -> None:
    """The real payload type (a NumPy array) survives save -> load."""
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    target = cache_engine._ssd_cache_path / "arr.zstd"
    cache_engine._save_to_disk(target, arr)

    got = cache_engine._load_from_disk(target)
    np.testing.assert_array_equal(np.asarray(got), arr)
