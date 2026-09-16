"""GUI test configuration: force the offscreen Qt platform before any QApplication."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def fake_handle_factory():
    """Factory for ``FitQueueController(handle_factory=...)``: returns
    :class:`tests.gui.ipc_fakes.FakeHandle`, the single shared WorkerHandle
    test double (see its docstring for why the six ad-hoc copies were
    consolidated here).
    """
    from tests.gui.ipc_fakes import FakeHandle

    return FakeHandle
