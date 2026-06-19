"""Root pytest configuration."""

import importlib.util


def pytest_addoption(parser, pluginmanager):
    """Block the pytest-qt plugin when PySide6 is absent (bare [dev] / CLI envs).

    pytest-qt raises QtBindingsNotFoundError during plugin init if it loads
    without a Qt binding; blocking it here (before init) lets GUI test modules
    self-skip via ``pytest.importorskip("PySide6")`` instead of crashing the run.
    Block by the registered pytest11 entry-point name ``pytest-qt`` (hyphenated) —
    ``set_blocked("pytestqt")`` would be a silent no-op (wrong name).
    """
    if importlib.util.find_spec("PySide6") is None:
        pluginmanager.set_blocked("pytest-qt")
