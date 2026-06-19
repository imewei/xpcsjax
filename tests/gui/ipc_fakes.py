"""Importable spawn-target fakes for the WorkerHandle tests.

Run inside spawned children, so this stays a plain module (no pytest, no
importorskip, no qtbot) importing nothing heavy.
"""

from __future__ import annotations

import time
from typing import Any

from xpcsjax.gui.ipc.job import FitJob
from xpcsjax.service.events import Finished, Started


def emit_started_then_finished(job: FitJob, q: Any) -> None:
    q.put(Started(run_id=job.run_id, seq=1, mode="m", settings_summary="s"))
    q.put(Finished(run_id=job.run_id, seq=2, result_path="/tmp/out"))


def exit_without_terminal(job: FitJob, q: Any) -> None:
    q.put(Started(run_id=job.run_id, seq=1, mode="m", settings_summary="s"))
    raise SystemExit(3)  # abnormal exit, no terminal event


def sleep_forever(job: FitJob, q: Any) -> None:
    q.put(Started(run_id=job.run_id, seq=1, mode="m", settings_summary="s"))
    time.sleep(120)
