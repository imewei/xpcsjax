"""Tests for the JAX-free Project/Dataset/FitRun model."""

from xpcsjax.gui.project.model import DONE, QUEUED, RUNNING, Project


def test_add_dataset_assigns_stable_unique_ids():
    p = Project()
    d1 = p.add_dataset("a.yaml")
    d2 = p.add_dataset("b.yaml", label="Run B")
    assert d1.dataset_id != d2.dataset_id
    assert d1.label  # auto-labeled from the config filename when not given
    assert d2.label == "Run B"
    assert p.dataset_by_id(d1.dataset_id) is d1


def test_add_run_is_append_only_and_queued():
    p = Project()
    d = p.add_dataset("a.yaml")
    r1 = p.add_run(d.dataset_id)
    r2 = p.add_run(d.dataset_id)
    assert [r.status for r in d.runs] == [QUEUED, QUEUED]
    assert r1.run_id != r2.run_id
    assert r1.created_at  # stamped
    found = p.run_by_id(r2.run_id)
    assert found is not None and found[0] is d and found[1] is r2


def test_set_run_status_updates_in_place():
    p = Project()
    d = p.add_dataset("a.yaml")
    r = p.add_run(d.dataset_id)
    p.set_run_status(r.run_id, RUNNING)
    assert r.status == RUNNING
    p.set_run_status(r.run_id, DONE, result_dir="/tmp/out", summary={"chi2": 1.0})
    assert r.status == DONE and r.result_dir == "/tmp/out" and r.summary == {"chi2": 1.0}


def test_unknown_ids_return_none():
    p = Project()
    assert p.dataset_by_id("nope") is None
    assert p.run_by_id("nope") is None
