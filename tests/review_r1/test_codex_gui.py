"""Red tests for the Codex review round 1, shard gui (xpcsjax/gui + xpcsjax/service).

One test per CONFIRMED claim; each is expected to FAIL until the defect is
fixed. gui-B2 (DESIGN) and gui-B14 (REJECTED: no in-tree caller) were removed.
"""

from __future__ import annotations

import copy
import errno
import os
import struct
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # before any PySide6 import

import numpy as np
import pytest


def _opt_result(parameters):
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    p = np.asarray(parameters, dtype=float)
    return OptimizationResult(
        parameters=p,
        uncertainties=np.zeros_like(p),
        covariance=np.eye(p.size),
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        convergence_status="converged",
        iterations=1,
        execution_time=0.0,
        device_info={},
    )


# --------------------------------------------------------------------------- B1
def test_gui_b1():
    """Persisted labels must follow the solver's canonical physics order, not
    the order the user happened to list ``active_parameters`` in."""
    from xpcsjax.config import ConfigManager
    from xpcsjax.optimization.nlsq.core import _get_physical_param_names
    from xpcsjax.service.persist import _resolve_parameter_names

    canonical = list(_get_physical_param_names("static_isotropic"))
    assert canonical == ["D0", "alpha", "D_offset"]  # solver's result.parameters tail
    cm = ConfigManager(
        config_override={
            "analysis_mode": "static_isotropic",
            "initial_parameters": {"active_parameters": ["alpha", "D0", "D_offset"]},
        }
    )
    result = SimpleNamespace(
        parameters=np.array([0.3, 1.0, 8000.0, -1.2, 50.0]), nlsq_diagnostics={}
    )
    names = _resolve_parameter_names(cm, result)
    assert names is not None
    assert names[-3:] == canonical


# --------------------------------------------------------------------------- B3
def test_gui_b3(monkeypatch):
    """A NaN requested angle matches nothing: per the documented contract the
    subsetter must fit all angles (or reject), never silently pick index 0."""
    import xpcsjax.service.data as svc_data

    base = {
        "c2_exp": np.arange(3 * 4 * 4, dtype=float).reshape(3, 4, 4),
        "phi_angles_list": np.array([0.0, 45.0, 90.0]),
    }
    monkeypatch.setattr(
        svc_data, "load_xpcs_data", lambda config_dict: {k: v.copy() for k, v in base.items()}
    )
    monkeypatch.setattr(svc_data, "apply_angle_filtering_for_optimization", lambda data, cm: data)
    cfg = {"analysis_mode": "laminar_flow", "data_type": "aps_old"}
    try:
        out = svc_data.load_dataset(
            SimpleNamespace(config=cfg, get_config=lambda: cfg), phi_subset=[float("nan")]
        )
    except ValueError:
        return  # rejecting the angle is also acceptable
    assert out["c2_exp"].shape[0] == 3, "phi=NaN silently selected the first angle"


# --------------------------------------------------------------------------- B4
def test_gui_b4():
    """NRMSE must be evaluated with the q the fit used (the data's), so it must
    not change when only the *config* q differs from the measured q."""
    import xpcsjax.service.fit_quality as fq
    from tests.optimization.test_phase5_standard_resolver import _laminar_cfg
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.homodyne_model import HomodyneModel

    n_t = 12
    cfg = _laminar_cfg("auto", n_t)
    phi = np.array([0.0])
    true = np.array([1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0])
    c2 = np.asarray(HomodyneModel(cfg.config).compute_c2(true, phi, contrast=0.3, offset=1.0))
    t = np.arange(n_t, dtype=np.float64) * 0.1
    data = {
        "phi_angles_list": phi,
        "c2_exp": c2,
        "t1": t,
        "t2": t,
        "wavevector_q_list": np.array([0.0237]),  # the q the fit actually uses
    }
    result = _opt_result([0.3, 1.0, *true])

    nrmse_same_q = fq.compute_fit_quality(result, data, cfg)["nrmse"]
    assert nrmse_same_q < 1e-6  # control: exact model at the data q

    other = copy.deepcopy(cfg.config)
    other["analyzer_parameters"]["scattering"]["wavevector_q"] = 2 * 0.0237
    cfg2 = ConfigManager(config_override=other)
    nrmse_cfg_q_differs = fq.compute_fit_quality(result, data, cfg2)["nrmse"]
    assert nrmse_cfg_q_differs == pytest.approx(nrmse_same_q, abs=1e-6)


# --------------------------------------------------------------------------- B5
def test_gui_b5(qtbot, tmp_path, monkeypatch):
    """A run persisted as 'running' has no worker after reopen; it must not
    block a new Run for that dataset forever."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QMessageBox

    from xpcsjax.gui.app import build_workbench
    from xpcsjax.gui.project.model import RUNNING, Project
    from xpcsjax.gui.project.persist import save_project

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("analysis_mode: static_isotropic\n", encoding="utf-8")
    p = Project()
    d = p.add_dataset(str(cfg))
    r = p.add_run(d.dataset_id)
    p.set_run_status(r.run_id, RUNNING)
    proj = tmp_path / "s.xpcsproj"
    save_project(p, proj)

    win, _ = build_workbench()
    qtbot.addWidget(win)
    win.open_project_from(proj)
    calls = []
    monkeypatch.setattr(win._queue, "enqueue", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    win._on_run()
    assert len(calls) == 1, "Run blocked by a stale persisted 'running' status"


# --------------------------------------------------------------------------- B7
def test_gui_b7(qtbot, tmp_path, monkeypatch):
    """A bundle load still in flight when the project is closed must not
    repopulate the cleared workbench."""
    pytest.importorskip("PySide6")
    from xpcsjax.gui.app import build_workbench
    from xpcsjax.gui.views.main_window_support import result_presenter as rp
    from xpcsjax.gui.viz_bundle import VizBundle

    held = []
    pool = SimpleNamespace(start=held.append)  # hold the task: never runs
    monkeypatch.setattr(rp.QThreadPool, "globalInstance", staticmethod(lambda: pool))

    win, _ = build_workbench()
    qtbot.addWidget(win)
    bundle = VizBundle(exp_c2=np.ones((1, 4, 4)))

    # Positive control: without a close, the late bundle does show the grid.
    win._show_result_with_bundle(None, str(tmp_path))
    assert len(held) == 1
    win._result_presenter._on_bundle_loaded(bundle, str(tmp_path))
    assert win._central_stack.currentIndex() == 1

    win._show_result_with_bundle(None, str(tmp_path))
    win.close_project()
    assert win._central_stack.currentIndex() == 0
    win._result_presenter._on_bundle_loaded(bundle, str(tmp_path))  # late arrival
    assert win._central_stack.currentIndex() == 0
    assert win._result_grid.section_count() == 0


# --------------------------------------------------------------------------- B8
def test_gui_b8(qtbot, tmp_path, monkeypatch):
    """A failed save (disk full mid-write) must not destroy the previous YAML."""
    pytest.importorskip("PySide6")
    from pathlib import Path

    from xpcsjax.gui.views.config_dialogs import ConfigTextEditorDialog

    cfg = tmp_path / "cfg.yaml"
    original = b"analysis_mode: static_isotropic\n"
    cfg.write_bytes(original)
    dlg = ConfigTextEditorDialog(cfg)
    qtbot.addWidget(dlg)
    assert dlg.load_error() is None
    dlg._editor.setPlainText("analysis_mode: laminar_flow\n")
    dlg.accept = lambda: None

    def _enospc_write_text(self, data, *a, **k):
        with open(self, "w") as fh:  # real open-for-write truncates, as write_text does
            fh.write(data[:3])
        raise OSError(errno.ENOSPC, "disk full")

    with monkeypatch.context() as m:
        m.setattr(Path, "write_text", _enospc_write_text)
        dlg._on_save()  # must not raise; surfaces the error in the dialog
    assert "save" in dlg._error_label.text().lower()
    assert cfg.read_bytes() == original


# --------------------------------------------------------------------------- B9
def test_gui_b9(tmp_path, monkeypatch):
    import xpcsjax.service.persist as persist

    real_replace = os.replace
    state = {"nested": False}

    def _replace(src, dst):
        if not state["nested"]:
            state["nested"] = True
            persist.save_results_json(_opt_result([1.0, 2.0]), tmp_path)  # writer B
        return real_replace(src, dst)

    monkeypatch.setattr(persist.os, "replace", _replace)
    assert persist.save_results_json(_opt_result([3.0, 4.0]), tmp_path).is_file()  # writer A


# -------------------------------------------------------------------------- B10
def test_gui_b10(tmp_path, monkeypatch):
    """merge_fitted_c2's contract is 'never raises' (best-effort enrichment)."""
    import xpcsjax.service.persist as persist

    primary = tmp_path / "nlsq_result.npz"
    fitted = tmp_path / "plots" / "simulated_data" / "c2_fitted_data.npz"
    np.savez(primary, parameters=np.array([1.0, 2.0]))
    fitted.parent.mkdir(parents=True)
    c2 = np.ones((2, 3, 3))
    np.savez(fitted, c2_exp=c2, c2_fitted=c2 * 0.9, residuals=c2 * 0.1)
    before = primary.read_bytes()

    def _enospc(*a, **k):
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(persist.tempfile, "NamedTemporaryFile", _enospc)
    assert persist.merge_fitted_c2(primary, fitted) is False
    assert primary.read_bytes() == before


# -------------------------------------------------------------------------- B11
@pytest.mark.parametrize("text", ["[]", "null", "42"])
def test_gui_b11(tmp_path, text):
    from xpcsjax.gui.project.persist import load_project

    bad = tmp_path / "bad.xpcsproj"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_project(bad)


# -------------------------------------------------------------------------- B12
def test_gui_b12(qtbot, tmp_path):
    """A DONE run whose result dir exists but holds no loadable summary must be
    flagged result_missing (open_project_from docstring: 'or whose summary
    will not load')."""
    pytest.importorskip("PySide6")
    from xpcsjax.gui.app import build_workbench
    from xpcsjax.gui.project.model import DONE, Project
    from xpcsjax.gui.project.persist import save_project

    p = Project()
    d = p.add_dataset(str(tmp_path / "cfg.yaml"), label="DS-A")
    r = p.add_run(d.dataset_id)
    result_dir = tmp_path / "partial_copy"
    result_dir.mkdir()  # exists, but nlsq_result.json is absent
    p.set_run_status(r.run_id, DONE, result_dir=str(result_dir))
    proj = tmp_path / "s.xpcsproj"
    save_project(p, proj)

    win, _ = build_workbench()
    qtbot.addWidget(win)
    win.open_project_from(proj)
    _, run = win._project.run_by_id(r.run_id)
    assert run.summary is None
    assert run.result_missing is True


# -------------------------------------------------------------------------- B13
def test_gui_b13(tmp_path):
    h5py = pytest.importorskip("h5py")
    from xpcsjax.gui.data_inspect import read_hdf5_metadata

    path = tmp_path / "d.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("optional", data=h5py.Empty("f8"))
        f.create_dataset("ordinary", data=[1.0])
    assert {i.name for i in read_hdf5_metadata(path)} == {"optional", "ordinary"}


# --------------------------------------------------------------------------- B6
def _partial_frame_then_die(job, q):
    """Fake worker: write only a frame header (announcing 64 KiB), then die hard."""
    os.write(q._writer.fileno(), struct.pack("!i", 65536))
    os._exit(9)


@pytest.mark.skipif(os.name != "posix", reason="POSIX pipe framing")
def test_gui_b6(qtbot, monkeypatch):
    """A worker that dies mid-frame must still yield a terminal Died event."""
    pytest.importorskip("PySide6")
    from xpcsjax.gui.ipc import handle as handle_mod
    from xpcsjax.gui.ipc.handle import WorkerHandle
    from xpcsjax.gui.ipc.job import FitJob
    from xpcsjax.service.events import Died

    monkeypatch.setattr(handle_mod, "run_worker", _partial_frame_then_die)
    h = WorkerHandle(FitJob(run_id="r1", config_path="c.yaml"))
    events = []
    h.event.connect(events.append)
    h.start()
    try:
        qtbot.waitUntil(lambda: not h.is_running(), timeout=20000)  # worker is dead
        qtbot.waitUntil(lambda: any(isinstance(e, Died) for e in events), timeout=5000)
    finally:
        # Unblock the reader no matter what: dropping the parent's write end
        # turns its blocked frame read into EOF.
        h._queue._writer.close()
        h._reader.wait(5000)
        h.shutdown()
