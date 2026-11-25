"""cli-B2 twin: the GUI worker must not merge plot arrays into a stale NPZ.

With ``output_format="json"`` the save writes no ``nlsq_result.npz``, so an NPZ
left over from an older run must not receive this run's fitted-c2 arrays.
"""

from __future__ import annotations

import queue as queue_mod
import sys
import types
from pathlib import Path

import pytest

from xpcsjax.gui.ipc.job import FitJob


def _install(monkeypatch, merges: list) -> None:
    mods = {
        "xpcsjax.service.config": types.ModuleType("xpcsjax.service.config"),
        "xpcsjax.service.data": types.ModuleType("xpcsjax.service.data"),
        "xpcsjax.service.fit": types.ModuleType("xpcsjax.service.fit"),
        "xpcsjax.service.persist": types.ModuleType("xpcsjax.service.persist"),
        "xpcsjax.service.plots": types.ModuleType("xpcsjax.service.plots"),
    }
    mods["xpcsjax.service.config"].load_config = lambda path, **_kw: types.SimpleNamespace(
        config={"analysis_mode": "laminar_flow"}
    )
    mods["xpcsjax.service.data"].load_dataset = lambda cm, **_kw: {}
    mods["xpcsjax.service.fit"].FitOverrides = lambda **kw: dict(kw)
    mods["xpcsjax.service.fit"].run_fit = lambda cm, d, **_kw: object()
    mods["xpcsjax.service.persist"].save_results = lambda *a, **k: None
    mods["xpcsjax.service.persist"].merge_fitted_c2 = lambda *a, **k: merges.append(a) or True
    mods["xpcsjax.service.plots"].generate_plots = lambda *a, **k: Path("plots")
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)


@pytest.mark.parametrize(
    ("output_format", "expect_merge"),
    [("json", False), ("npz", True), ("both", True)],
)
def test_worker_merges_plots_only_when_npz_written(
    monkeypatch, tmp_path, output_format, expect_merge
):
    from xpcsjax.gui.ipc.worker import run_worker

    merges: list = []
    _install(monkeypatch, merges)
    job = FitJob(
        run_id="r1", config_path="c.yaml", output_dir=str(tmp_path), output_format=output_format
    )
    run_worker(job, queue_mod.Queue())
    assert bool(merges) is expect_merge
