"""Tests for DataInspectDialog — the wiring for the previously-orphaned
data_inspect.py module (metadata listing + C2 preview) into the GUI."""

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
h5py = pytest.importorskip("h5py")

from xpcsjax.gui.views.data_inspect_dialog import DataInspectDialog  # noqa: E402


def _make_h5(path):
    with h5py.File(path, "w") as f:
        f.create_dataset("c2", data=np.random.default_rng(0).random((2, 32, 32)))


def test_lists_datasets(qtbot, tmp_path):
    p = tmp_path / "d.h5"
    _make_h5(p)
    dlg = DataInspectDialog(p)
    qtbot.addWidget(dlg)
    assert dlg.dataset_count() >= 1
    assert dlg.load_error() is None


def test_missing_file_shows_error_not_crash(qtbot, tmp_path):
    dlg = DataInspectDialog(tmp_path / "does_not_exist.h5")
    qtbot.addWidget(dlg)
    assert dlg.dataset_count() == 0
    assert dlg.load_error() is not None


def test_preview_renders_selected_dataset(qtbot, tmp_path):
    p = tmp_path / "d.h5"
    _make_h5(p)
    dlg = DataInspectDialog(p)
    qtbot.addWidget(dlg)
    dlg._dataset_list.setCurrentRow(0)
    dlg._on_preview()
    assert dlg.load_error() is None
    assert dlg._preview.has_image()
