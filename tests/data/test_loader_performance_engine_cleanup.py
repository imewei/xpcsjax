"""Regression test: XPCSDataLoader.close() must shut down its memory_manager.

``xpcsjax.data.performance_engine`` (and the loader's ``performance_engine``
attribute, always ``None`` in production) was removed in the 2026-09-15
review (finding B1) — it had zero production callers. The memory_manager
shutdown contract these tests originally shared a file with is real and
stays covered here.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from xpcsjax.data.xpcs_loader import XPCSDataLoader


def _bare_loader() -> XPCSDataLoader:
    """Bypass __init__ to avoid needing a YAML config / real dataset on disk."""
    return XPCSDataLoader.__new__(XPCSDataLoader)


def test_close_is_idempotent_and_safe_with_no_components():
    loader = _bare_loader()
    loader.memory_manager = None

    loader.close()
    loader.close()  # must not raise on a second call


def test_context_manager_closes_on_exit():
    loader = _bare_loader()
    mm = MagicMock()
    loader.memory_manager = mm

    with loader:
        mm.shutdown.assert_not_called()

    mm.shutdown.assert_called_once()


def test_close_calls_memory_manager_shutdown():
    loader = _bare_loader()
    mm = MagicMock()
    loader.memory_manager = mm

    loader.close()

    mm.shutdown.assert_called_once()
    assert loader.memory_manager is None


def test_performance_engine_attribute_no_longer_exists():
    """B1: the attribute (always None) was removed along with the module."""
    loader = _bare_loader()
    assert not hasattr(loader, "performance_engine")
