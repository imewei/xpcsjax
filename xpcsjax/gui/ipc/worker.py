"""The spawn-target child entrypoint: run a fit and stream events.

Module-level imports are JAX-free (the GUI process imports this module just to
reference :func:`run_worker` as the spawn target). All ``service.*`` imports —
which pull in JAX — live inside :func:`run_worker`, executed only in the child.
"""

from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path
from typing import Any

from xpcsjax.gui.ipc.diagnostics import layer_status_from_diagnostics
from xpcsjax.gui.ipc.emitter import EventEmitter
from xpcsjax.gui.ipc.job import FitJob
from xpcsjax.gui.ipc.log_capture import BannerLogHandler, QueueLogHandler
from xpcsjax.service.events import Failed, Finished, LayerStatus, LogLine


def run_worker(job: FitJob, event_queue: Any) -> None:
    """Run ``job`` to completion, streaming events onto ``event_queue``.

    Emits ``Started`` (via ``run_fit``'s ``on_event``), ``LogLine``s (via a
    logging handler on the ``xpcsjax`` logger), and exactly one terminal event
    (``Finished`` on success, ``Failed`` on any exception). The parent
    synthesizes ``Died`` if the process exits without a terminal event.
    """
    # POSIX: own a process group so the parent can tear down any grandchildren.
    if hasattr(os, "setpgrp"):
        try:
            os.setpgrp()
        except OSError:  # pragma: no cover — platform dependent
            pass

    emitter = EventEmitter(event_queue, job.run_id)
    handler = QueueLogHandler(emitter)
    banner_handler = BannerLogHandler(emitter)
    root = logging.getLogger("xpcsjax")
    root.setLevel(logging.INFO)  # so INFO banners + log lines actually reach the handlers
    root.addHandler(handler)
    root.addHandler(banner_handler)
    try:
        from xpcsjax.service.config import load_config
        from xpcsjax.service.data import load_dataset
        from xpcsjax.service.fit import FitOverrides, run_fit
        from xpcsjax.service.persist import merge_fitted_c2, save_results

        config_manager = load_config(job.config_path, output_dir=job.output_dir)
        phi_subset = list(job.phi_subset) if job.phi_subset else None
        data = load_dataset(config_manager, phi_subset=phi_subset)
        overrides = FitOverrides(**job.overrides) if job.overrides else None

        result = run_fit(
            config_manager, data, overrides=overrides, run_id=job.run_id, on_event=emitter
        )

        out_dir = Path(job.output_dir) if job.output_dir else None
        if out_dir is not None:
            # F9 (atomic result write) is satisfied transitively: service.persist now
            # writes each file via temp + os.replace (Plan B2 Step 2b), and the
            # Finished event below is emitted only AFTER save_results returns — so the
            # GUI never loads a half-written result. No extra worker code is needed.
            save_results(result, out_dir, job.output_format, config_manager, None)
            if job.make_plots:
                from xpcsjax.service.plots import generate_plots

                plots_result = generate_plots(result, data, config_manager, out_dir / "plots")
                if plots_result is not None:
                    # Mirrors cli/commands.py's post-plot merge: the "simulated"
                    # plot family writes c2_exp/c2_fitted/residuals to
                    # plots/simulated_data/c2_fitted_data.npz; fold those into
                    # the primary nlsq_result.npz too, best-effort.
                    merge_fitted_c2(
                        out_dir / "nlsq_result.npz",
                        plots_result / "simulated_data" / "c2_fitted_data.npz",
                    )
                if plots_result is None:
                    # generate_plots swallows its own errors and returns None on
                    # failure (documented contract) -- without this check the run
                    # still reports a plain Finished with no visible sign that
                    # plotting failed, aside from a WARNING buried in the log
                    # stream. Surface it explicitly.
                    emitter.emit(
                        LogLine(
                            run_id="",
                            seq=0,
                            level="ERROR",
                            msg=(
                                "Plot generation failed; fit result was saved but no "
                                "plots were produced (see log above for details)."
                            ),
                        )
                    )

        diagnostics = getattr(result, "nlsq_diagnostics", None)
        cfg = getattr(config_manager, "config", None) or {}  # config may be None
        emitter.emit(
            LayerStatus(
                run_id="",
                seq=0,
                layers=layer_status_from_diagnostics(diagnostics),
                mode=str(cfg.get("analysis_mode", "")),
            )
        )
        emitter.emit(Finished(run_id="", seq=0, result_path=str(out_dir) if out_dir else ""))
    except BaseException:  # noqa: BLE001 — report ANY failure as a terminal event
        emitter.emit(Failed(run_id="", seq=0, traceback=traceback.format_exc()))
    finally:
        root.removeHandler(handler)
        root.removeHandler(banner_handler)
