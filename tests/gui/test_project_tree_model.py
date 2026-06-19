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


def test_dead_paths_are_flagged_missing_in_tree(qtbot):
    # Spec §8 dead paths: a run with a gone result_dir and a dataset with a gone
    # config_path must be surfaced as clearly-flagged "missing" entries, not
    # rendered as normal rows.
    p = Project()
    d = p.add_dataset("gone.yaml", label="DS-A")
    d.config_missing = True
    r = p.add_run(d.dataset_id)
    p.set_run_status(r.run_id, DONE)
    p.run_by_id(r.run_id)[1].result_missing = True
    model = ProjectTreeModel()
    model.rebuild(p)

    ds_item = model.item(0)
    assert "config missing" in ds_item.text()
    assert "result missing" in ds_item.child(0).text()
