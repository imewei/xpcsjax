"""Smoke test: xpcs_loader module imports cleanly with no homodyne references."""

import xpcsjax.data.xpcs_loader as loader


def test_module_imports():
    assert loader is not None


def test_load_xpcs_data_callable():
    assert callable(loader.load_xpcs_data)
