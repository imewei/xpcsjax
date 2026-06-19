"""pytest-qt tests for the Project -> Qt tree mirror."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402

from xpcsjax.gui.project.model import DONE, Project  # noqa: E402
from xpcsjax.gui.project.tree_model import ProjectTreeModel  # noqa: E402


def test_rebuild_mirrors_datasets_and_runs(qtbot):
    p = Project()
    d = p.add_dataset("a.yaml", label="DS-A")
    p.add_run(d.dataset_id)
    p.add_run(d.dataset_id)
    model = ProjectTreeModel()
    model.rebuild(p)

    assert model.rowCount() == 1  # one dataset
    ds_item = model.item(0)
    assert ds_item.text() == "DS-A"
    assert ds_item.data(Qt.ItemDataRole.UserRole) == d.dataset_id
    assert ds_item.rowCount() == 2  # two runs


def test_update_run_refreshes_status_label(qtbot):
    p = Project()
    d = p.add_dataset("a.yaml")
    r = p.add_run(d.dataset_id)
    model = ProjectTreeModel()
    model.rebuild(p)
    p.set_run_status(r.run_id, DONE)
    model.update_run(p, r.run_id)

    run_item = model.item(0).child(0)
    assert DONE in run_item.text()
    assert run_item.data(Qt.ItemDataRole.UserRole) == r.run_id
