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


def test_preview_uses_stored_name_not_display_text(qtbot, tmp_path):
    """HDF5 names may contain spaces; the label they're rendered into is not a
    parseable source for the dataset key."""
    p = tmp_path / "d.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("my data/c2", data=np.random.default_rng(0).random((2, 8, 8)))
    dlg = DataInspectDialog(p)
    qtbot.addWidget(dlg)
    dlg._dataset_list.setCurrentRow(0)
    dlg._on_preview()
    assert dlg.load_error() is None
    assert dlg._preview.has_image()


def test_empty_leading_axis_shows_error_not_crash(qtbot, tmp_path):
    """A 3-D dataset with a zero-length first axis makes the auto-mode read raise
    IndexError — it must land on the error label, not escape the button slot."""
    p = tmp_path / "d.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("c2", shape=(0, 4, 4), dtype="f8")
    dlg = DataInspectDialog(p)
    qtbot.addWidget(dlg)
    dlg._dataset_list.setCurrentRow(0)
    dlg._on_preview()
    assert dlg.load_error() is not None


def test_corrupt_file_error_shown_not_crash(qtbot, tmp_path, monkeypatch):
    """A corrupt-but-signature-valid HDF5 file can raise RuntimeError/KeyError/
    ValueError from h5py, not just OSError — the dialog must degrade to the
    error label, not propagate the exception through __init__."""
    import xpcsjax.gui.views.data_inspect_dialog as dlg_mod

    p = tmp_path / "d.h5"
    _make_h5(p)

    def _raise(*_a, **_k):
        raise RuntimeError("Unable to synchronously visit object (bad object header)")

    monkeypatch.setattr(dlg_mod, "read_hdf5_metadata", _raise)
    dlg = DataInspectDialog(p)
    qtbot.addWidget(dlg)
    assert dlg.dataset_count() == 0
    assert dlg.load_error() is not None
