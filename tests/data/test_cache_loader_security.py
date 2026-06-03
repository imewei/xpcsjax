"""Defense-in-depth regression tests for the trusted-cache loader.

The /double-check security pass flagged ``performance_engine.py:_load_from_disk``
as the only serialized-deserialization site left in the package. The original
mitigation was just a documented threat model ("we only load files this class
wrote"). Phase 5 hardening added three pre-deserialization gates — path
containment, ownership, and mode — so a compromised cache directory cannot
escalate to arbitrary code execution. These tests pin each gate.

If any of these tests start failing, **do not loosen the assertion**. The
gates are the security boundary; a regression here means the threat model is
no longer enforced.

Tests deliberately avoid importing ``pickle`` directly — fixtures are written
through the engine's own ``_save_to_disk`` so the test mirrors the real
write-then-read flow that hardens the loader.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def cache_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Construct a MultiLevelCache pinned at ``tmp_path``.

    Overrides ``XDG_CACHE_HOME`` so the cache writes under ``tmp_path``
    instead of the user's real cache directory. ``MultiLevelCache`` reads
    that env var in its ``__init__`` to build ``_cache_base_path``.
    """
    from xpcsjax.data.performance_engine import MultiLevelCache

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    # Tiny budgets — we're not exercising eviction here, just the loader gate.
    return MultiLevelCache(memory_cache_mb=1.0, ssd_cache_mb=1.0, hdd_cache_mb=1.0)


def _write_via_engine(engine: Any, key: str, item: Any) -> Path:
    """Write a cache item through the engine's own save path and return its disk path.

    Using ``_save_to_disk`` (rather than constructing the on-disk format by
    hand) guarantees the test exercises the same write contract the loader
    expects — including 0o600 mode, zstd framing, and the
    ``np.save(allow_pickle=False)`` serialization (SEC-1). The disk tier only
    stores numeric arrays, so fixtures use arrays rather than dicts.
    """
    target = engine._ssd_cache_path / f"{key}.zstd"
    engine._save_to_disk(target, item)
    return target


# ---------------------------------------------------------------------------
# Gate (0): the happy path — write via engine, read via engine, item survives.
# ---------------------------------------------------------------------------


def test_round_trip_through_engine_loads_ok(cache_engine: Any) -> None:
    """Sanity: a file written through ``_save_to_disk`` loads cleanly.

    If this fails, the hardening broke the happy path — likely the mode or
    ownership check is too strict for the file the engine just wrote.
    """
    payload = np.array([1.0, 2.0, 3.0])
    target = _write_via_engine(cache_engine, "ok", payload)
    got = cache_engine._load_from_disk(target)
    np.testing.assert_array_equal(np.asarray(got), payload)


# ---------------------------------------------------------------------------
# Gate (1): path containment.
# ---------------------------------------------------------------------------


def test_symlink_escape_outside_cache_root_is_refused(
    cache_engine: Any, tmp_path: Path
) -> None:
    """Path-containment gate: a symlink pointing outside the cache root fails.

    Simulates a poisoned cache that tries to redirect the loader to an
    attacker-controlled file outside the trusted cache root.
    """
    # Write a legitimate cache file, then symlink-redirect a fake key to it
    # in a location outside the cache root.
    legit = _write_via_engine(cache_engine, "real", np.array([1, 2, 3]))
    outside = tmp_path / "elsewhere" / "outside.zstd"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(legit.read_bytes())
    outside.chmod(0o600)

    decoy = cache_engine._ssd_cache_path / "decoy.zstd"
    decoy.symlink_to(outside)

    with pytest.raises(OSError, match="outside cache root"):
        cache_engine._load_from_disk(decoy)


def test_traversal_path_is_refused(cache_engine: Any, tmp_path: Path) -> None:
    """Path-containment gate: an explicit ``..``-traversal path fails.

    Even if a key-derivation bug let user input flow through, ``resolve()``
    + ``relative_to()`` catches the escape before deserialization.
    """
    # Write a legitimate file outside the cache root the traversal points at.
    outside = tmp_path / "outside.zstd"
    legit = _write_via_engine(cache_engine, "real", np.array([1, 2, 3]))
    outside.write_bytes(legit.read_bytes())
    outside.chmod(0o600)

    traversal = cache_engine._ssd_cache_path / ".." / ".." / "outside.zstd"
    with pytest.raises(OSError, match="outside cache root"):
        cache_engine._load_from_disk(traversal)


# ---------------------------------------------------------------------------
# Gate (3): mode. (Gate 2 — ownership — needs a different uid we can't easily
# fabricate in CI; covered by the documentation/contract and the mode check.)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32" or not hasattr(os, "getuid"),
    reason="POSIX ownership/mode semantics — Windows mode bits don't apply",
)
def test_world_writable_mode_is_refused(cache_engine: Any) -> None:
    """Mode gate: a file with world-write bits is refused.

    ``_save_to_disk`` writes 0o600. A cache file ``chmod 666``'d after
    creation can no longer be trusted — refusing to load is the safe default.
    """
    target = _write_via_engine(cache_engine, "world_writable", np.array([1, 2, 3]))
    target.chmod(0o666)  # world-writable — the gate must reject this

    with pytest.raises(OSError, match=r"group/world access"):
        cache_engine._load_from_disk(target)


@pytest.mark.skipif(
    sys.platform == "win32" or not hasattr(os, "getuid"),
    reason="POSIX ownership/mode semantics — Windows mode bits don't apply",
)
def test_group_readable_mode_is_refused(cache_engine: Any) -> None:
    """Mode gate: even read-only group bits are refused.

    Anything wider than 0o600 means *something* changed the mode after
    ``_save_to_disk`` wrote it — refuse to load.
    """
    target = _write_via_engine(cache_engine, "group_readable", np.array([1, 2, 3]))
    target.chmod(0o640)  # owner rw, group r — still a mode-drift signal

    with pytest.raises(OSError, match=r"group/world access"):
        cache_engine._load_from_disk(target)


# ---------------------------------------------------------------------------
# Edge case: missing file.
# ---------------------------------------------------------------------------


def test_missing_file_raises_oserror(cache_engine: Any) -> None:
    """A nonexistent cache path raises OSError before hitting deserialization.

    Without an explicit stat() check, the loader would leak a
    ``FileNotFoundError`` from inside the read — this pin keeps that surface
    closed and the error message coherent.
    """
    target = cache_engine._ssd_cache_path / "nope.zstd"
    with pytest.raises(OSError):
        cache_engine._load_from_disk(target)
