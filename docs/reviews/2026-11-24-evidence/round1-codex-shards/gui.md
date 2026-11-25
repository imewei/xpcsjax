# Codex review — shard gui
## Files read

xpcsjax/gui/\_\_init\_\_.py — 8 — full  
xpcsjax/gui/app.py — 108 — full  
xpcsjax/gui/comparison.py — 109 — full  
xpcsjax/gui/controllers/\_\_init\_\_.py — 3 — full  
xpcsjax/gui/controllers/fit_queue.py — 321 — full  
xpcsjax/gui/data_inspect.py — 123 — full  
xpcsjax/gui/error_presenter.py — 47 — full  
xpcsjax/gui/export.py — 83 — full  
xpcsjax/gui/ipc/\_\_init\_\_.py — 3 — full  
xpcsjax/gui/ipc/diagnostics.py — 62 — full  
xpcsjax/gui/ipc/emitter.py — 58 — full  
xpcsjax/gui/ipc/handle.py — 375 — full  
xpcsjax/gui/ipc/job.py — 26 — full  
xpcsjax/gui/ipc/log_capture.py — 49 — full  
xpcsjax/gui/ipc/worker.py — 117 — full  
xpcsjax/gui/project/\_\_init\_\_.py — 3 — full  
xpcsjax/gui/project/model.py — 105 — full  
xpcsjax/gui/project/persist.py — 127 — full  
xpcsjax/gui/project/tree_model.py — 93 — full  
xpcsjax/gui/result_loader.py — 99 — full  
xpcsjax/gui/theme.py — 498 — full  
xpcsjax/gui/views/\_\_init\_\_.py — 3 — full  
xpcsjax/gui/views/config_dialogs.py — 320 — full  
xpcsjax/gui/views/data_inspect_dialog.py — 172 — full  
xpcsjax/gui/views/error_dialog.py — 35 — full  
xpcsjax/gui/views/inspector.py — 128 — full  
xpcsjax/gui/views/main_window.py — 746 — full  
xpcsjax/gui/views/main_window_support/\_\_init\_\_.py — 1 — full  
xpcsjax/gui/views/main_window_support/project_dialog_handler.py — 189 — full  
xpcsjax/gui/views/main_window_support/result_presenter.py — 201 — full  
xpcsjax/gui/views/main_window_support/run_controller.py — 139 — full  
xpcsjax/gui/views/main_window_support/status_manager.py — 73 — full  
xpcsjax/gui/views/plots/\_\_init\_\_.py — 1 — full  
xpcsjax/gui/views/plots/grid.py — 296 — full  
xpcsjax/gui/views/plots/helpers.py — 111 — full  
xpcsjax/gui/views/plots/maps.py — 170 — full  
xpcsjax/gui/views/plots/residuals.py — 102 — full  
xpcsjax/gui/views/plots/squares.py — 57 — full  
xpcsjax/gui/views/plots_view.py — 31 — full  
xpcsjax/gui/views/project_panel.py — 148 — full  
xpcsjax/gui/views/raster.py — 47 — full  
xpcsjax/gui/viz_bundle.py — 71 — full  
xpcsjax/service/\_\_init\_\_.py — 20 — full  
xpcsjax/service/config.py — 216 — full  
xpcsjax/service/data.py — 238 — full  
xpcsjax/service/events.py — 107 — full  
xpcsjax/service/fit.py — 213 — full  
xpcsjax/service/fit_quality.py — 225 — full  
xpcsjax/service/persist.py — 623 — full  
xpcsjax/service/plots.py — 100 — full

## Bugs

### gui-B1: Persistence can label fitted homodyne parameters with the wrong names

- Location: xpcsjax/service/persist.py:247
- Severity: Required
- Failure scenario: A `static_isotropic` fit with `active_parameters: [alpha, D0, D_offset]` returns the canonical physics tail `[D0, alpha, D_offset]`. JSON and NPZ label those first two values in reverse order. A restricted active list can also make retained fixed parameters appear under generic or scaling labels.
- Root cause: `_resolve_parameter_names` uses the configured active list as the ordered result layout. The solver derives the tail order from the registry and retains inactive physics slots in the returned vector.
- Checked: `ParameterManager.get_active_parameters` preserves the explicit list order. `test_restricted_active_parameters_real_fit` checks a retained physics slot. The persistence tests use canonical, full-length fake active lists. Both writers call the same resolver.
- Red test: With a real `ConfigManager(config_override={"analysis_mode": "static_isotropic", "initial_parameters": {"active_parameters": ["alpha", "D0", "D_offset"]}})` and a result with `parameters=np.array([.3, 1., 8000., -1.2, 50.])` and empty diagnostics, assert `_resolve_parameter_names(cm, result)[-3:] == ["D0", "alpha", "D_offset"]`.
- Confidence: high

### gui-B2: Explicit angle selection leaves q values attached to discarded angles

- Location: xpcsjax/service/data.py:92
- Severity: Required
- Failure scenario: An APS-old load returns φ `[0, 45]` with q `[0.005, 0.006]`. With `--phi 45`, the homodyne fit receives only the 45° correlation surface but still uses `0.005` as its scalar q.
- Root cause: `_subset_data_by_phi` applies the selected indices to C₂ and φ aliases only. The loader produces an aligned q value per APS-old `(q, φ)` pair, and the homodyne normalizer takes the first q value from the returned list.
- Checked: `hdf5_readers.py` returns `filtered_dqlist` and `filtered_dphilist` from the same indices. The subsequent angle-filtering adapter preserves q unchanged. `test_load_dataset_subsets_by_phi` omits the q array.
- Red test: Extend `test_load_dataset_subsets_by_phi` with `base["wavevector_q_list"] = np.array([.005, .006, .007])` and assert `out["wavevector_q_list"].tolist() == [.006]`.
- Confidence: high

### gui-B3: `--phi nan` silently selects the first angle

- Location: xpcsjax/service/data.py:77
- Severity: Required
- Failure scenario: With a valid dataset containing φ `[0, 45, 90]`, `--phi nan` proceeds with only the 0° matrix.
- Root cause: The CLI parses `nan` as a float. The subsetter turns every distance into NaN, `argmin` yields index 0, and `NaN > tol` is false, so the index passes the match guard.
- Checked: `args_parser.py` uses `type=float` for `--phi`, and `validate_args` has no finite check for that option. `cli/data_pipeline.py` forwards it to `load_dataset`. Loader checks cover dataset angles, and service tests cover finite requests and empty angle arrays.
- Red test: In `test_load_dataset_subsets_by_phi`, call `load_dataset` with `phi_subset=[float("nan")]` and assert the call raises `ValueError` identifying the invalid angle.
- Confidence: high

### gui-B4: Homodyne NRMSE can measure a different model from the one fitted

- Location: xpcsjax/service/fit_quality.py:137
- Severity: Required
- Failure scenario: A requested config q of `0.005` selects a measured q of `0.006`. A homodyne result that predicts the data exactly at `0.006` can report nonzero NRMSE.
- Root cause: The homodyne solver takes q from loaded data. `compute_fit_quality` supplies the original config to the `CombinedModel` plotting evaluator, which reads q from `analyzer_parameters.scattering.wavevector_q`.
- Checked: The HDF5 readers can select the nearest measured q, `core._normalize_data_to_object` extracts `wavevector_q_list[0]`, and the wrapper uses `data.q`. `ConfigManager.get_model` returns `CombinedModel` for homodyne. The service's laminar end-to-end test uses identical configured and measured q values.
- Red test: In `tests/service/test_fit_quality.py`, generate exact C₂ with the existing laminar fixture's model and its original q, set `data["wavevector_q_list"]` to that original q, and use a result carrying the generating parameters in canonical order. Record NRMSE, double only the config q, and assert `compute_fit_quality(result, data, cm)["nrmse"] == pytest.approx(original_nrmse)`.
- Confidence: high

### gui-B5: Reloading a project saved during a fit can permanently block Run for that dataset

- Location: xpcsjax/gui/project/persist.py:117
- Severity: Required
- Failure scenario: Save a project while a run is `queued`, `starting`, or `running`, then reopen it in a fresh workbench. Run reports an existing job, while Cancel has no corresponding job to cancel.
- Root cause: Persistence restores live queue statuses as ordinary history. Opening a project neither restores those workers nor changes their statuses, and `RunController.on_run` uses the historical statuses to block duplicates.
- Checked: `save_project_to` permits saving during a run. `open_project_from` restores the model and never enqueues old jobs. `FitQueueController.cancel` only acts on pending or active handles. Project round-trip tests use a completed run or no runs, and the duplicate-run test covers a live session only.
- Red test: Using `qtbot`, create a `Project` with one dataset and a `RUNNING` run, save it, and open it in a fresh `MainWindow`. Spy on `_queue.enqueue`, stub `QMessageBox.information`, call `win._on_run()`, and assert one new job was enqueued.
- Confidence: high

### gui-B6: A partial worker event can leave the IPC reader blocked after the worker dies

- Location: xpcsjax/gui/ipc/handle.py:69
- Severity: Required
- Failure scenario: A worker is killed while its queue feeder is sending a long log line or traceback. The reader receives a partial frame and can stay blocked indefinitely. The queue slot never gets a terminal event, and interruption cannot stop the blocked read.
- Root cause: On POSIX, `multiprocessing.Queue.get(timeout=...)` times out waiting for readability. After that it performs a blocking read of the whole frame. The parent's queue still holds a writer endpoint, so the worker's death alone does not produce EOF. The reader checks process death and interruption only after control returns from `get`.
- Checked: `WorkerHandle.start` retains both queue endpoints in the parent. CPython's `multiprocessing/queues.py` and `connection.py` show the poll and full-frame read. Worker tests cover normal death, cancellation, and an artificially timed-out join, with no partial-frame case.
- Red test: On POSIX, create a real multiprocessing queue and `_ReaderThread(q, SimpleNamespace(is_alive=lambda: False, exitcode=-9), "r1")`. Write only `struct.pack("!i", 65536)` to `q._writer.fileno()` using `os.write`, start the reader, and assert `reader.wait(2500)`. In `finally`, close `q._writer` and join the reader so the test cleans up even when red.
- Confidence: high

### gui-B7: Closing or replacing a project can bring back its old plots

- Location: xpcsjax/gui/views/main_window.py:645
- Severity: Required
- Failure scenario: Select a completed run with a large plot bundle and close the project while it loads. After the close clears the central panel, the old bundle finishes loading and switches the empty workbench back to that run's plots.
- Root cause: `_reset_run_view_state` clears widgets directly and leaves `ResultPresenter._pending_result_dir` intact. The completion slot still accepts the old result directory.
- Checked: Both Close and Open call the reset method. The presenter invalidates pending requests on text or error presentation, which the reset bypasses. The current stale-load test supersedes one request with a different directory. The close-project test has no pending load.
- Red test: With `qtbot`, stub `QThreadPool.start` to hold the task, call `win._show_result_with_bundle(None, str(tmp_path))`, then `win.close_project()`. Deliver `win._result_presenter._on_bundle_loaded(VizBundle(exp_c2=np.ones((1, 4, 4))), str(tmp_path))` and assert the central page index and grid section count are both zero.
- Confidence: high

### gui-B8: A failed config save can destroy the previous YAML

- Location: xpcsjax/gui/views/config_dialogs.py:295
- Severity: Required
- Failure scenario: Saving over an existing YAML file when the filesystem runs out of space after opening it leaves the file truncated or empty. The dialog displays an error, but the previous config is already lost.
- Root cause: `save()` opens the target directly with `Path.write_text`. Truncation happens before the write and flush can succeed. The exception handler only updates the dialog.
- Checked: Edit Config invokes this dialog from `ProjectDialogHandler`, and `_on_save` checks syntax before calling `save`. `test_config_text_editor_surfaces_write_failure` fails during `mkdir`, before the original file can be opened. Project persistence has a separate test for preserving the old file across a failed atomic save.
- Red test: Extend the config editor write-failure test with an existing YAML file. After loading it, patch `Path.write_text` with a stub that opens its passed path in `"w"`, writes a short prefix, and raises `OSError(errno.ENOSPC, "disk full")`. Call `_on_save()` and assert the original YAML bytes are unchanged.
- Confidence: high

### gui-B9: Concurrent JSON result saves share the same staging file

- Location: xpcsjax/service/persist.py:334
- Severity: Required
- Failure scenario: Two CLI fits choose the same output directory. The second writer replaces `nlsq_result.json.tmp` after the first writer stages its payload. The first writer can then fail on a missing staged file or publish the second fit's payload.
- Root cause: `save_results_json` derives its temporary filename solely from the destination name. `os.replace` makes the rename atomic, but both writers still share the staging file.
- Checked: `cli.commands` sends the resolved output directory directly to `save_results`. GUI per-run namespaces protect the normal desktop path, but the CLI has no corresponding lock. NPZ uses a unique temporary file. JSON persistence tests use one writer.
- Red test: Patch `persist.os.replace` so that on the outer save's first call it runs a second `save_results_json(result_b, tmp_path)` before invoking the real replace. Use a one-shot flag so the inner replace proceeds normally. Call `save_results_json(result_a, tmp_path)` and assert both saves return successfully.
- Confidence: high

### gui-B10: Fitted-surface enrichment can report an already-saved fit as failed

- Location: xpcsjax/service/persist.py:606
- Severity: Required
- Failure scenario: The result and plots have been written, but temporary-file creation for the NPZ merge fails because the filesystem is now full or read-only. The GUI worker emits `Failed` even though the fit artifacts were saved. The CLI can exit with an error after saving a converged result.
- Root cause: `merge_fitted_c2` promises to return `False` on enrichment errors, but acquires its temporary file before the write exception handler. Neither caller locally guards the merge.
- Checked: The worker's outer exception handler sends `Failed`. `cli.commands` guards plot dispatch and calls the merge afterward. Existing merge tests cover missing files, freshness, and successful writes.
- Red test: Create valid inputs with the helpers used by `test_merge_fitted_c2_happy_path`. Then patch `persist.tempfile.NamedTemporaryFile` to raise `OSError(errno.ENOSPC, "disk full")`. Assert `merge_fitted_c2(primary, fitted) is False` and the primary bytes are unchanged.
- Confidence: high

### gui-B11: Non-object project JSON bypasses the Open Project error dialog

- Location: xpcsjax/gui/project/persist.py:103
- Severity: Minor
- Failure scenario: A `.xpcsproj` containing valid JSON `[]` or `null` causes an unhandled `AttributeError` when opened.
- Root cause: The schema check detects a non-dict, then the error message calls `payload.get` on that same value before `ValueError` can be raised.
- Checked: `ProjectDialogHandler.on_open_project` catches `ValueError` and `OSError`. Nearby tests cover invalid JSON and an object with a bad schema.
- Red test: Write `[]` to `tmp_path / "bad.xpcsproj"` and assert `load_project(path)` raises `ValueError` matching `"schema"`.
- Confidence: high

### gui-B12: Reopened runs with a missing or unreadable summary still appear as normal completed runs

- Location: xpcsjax/gui/views/main_window.py:619
- Severity: Minor
- Failure scenario: A completed run's directory survives a partial copy, but `nlsq_result.json` is absent or invalid. After reopening, the sidebar omits its “result missing” marker.
- Root cause: `open_project_from` sets `result_missing` when the directory is absent or the summary loader raises. The loader normally returns `None` on missing or invalid JSON, and that return leaves the flag false.
- Checked: `load_result_summary` explicitly returns `None` for these cases. The tree model relies on the flag. `test_open_tolerates_deleted_result_dir` uses a nonexistent directory, and result-loader tests exercise malformed JSON separately.
- Red test: Repeat `test_open_tolerates_deleted_result_dir` with an existing empty result directory. Assert `run.summary is None and run.result_missing`.
- Confidence: high

### gui-B13: A valid HDF5 null dataset aborts the data inspector

- Location: xpcsjax/gui/data_inspect.py:29
- Severity: Minor
- Failure scenario: An otherwise readable HDF5 file contains an optional dataset created with `h5py.Empty("f8")`. Opening Inspect Data File raises `TypeError` before the dataset list can be populated.
- Root cause: h5py exposes a null dataspace as `shape=None`. The metadata callback calls `tuple(obj.shape)` unconditionally.
- Checked: `DataInspectDialog._load` catches HDF5 and value errors but lets `TypeError` escape. Metadata tests cover ordinary arrays, and dialog error tests cover missing or corrupt files.
- Red test: Create an HDF5 file with `f.create_dataset("optional", data=h5py.Empty("f8"))` and `f.create_dataset("ordinary", data=[1.0])`. Assert the names returned by `read_hdf5_metadata(path)` are `{"optional", "ordinary"}`.
- Confidence: high

### gui-B14: Malformed parameter names make config validation raise

- Location: xpcsjax/service/config.py:190
- Severity: Minor
- Failure scenario: A caller validates parsed YAML containing `initial_parameters: {parameter_names: 42, values: [1]}` and receives `TypeError`. A sequence containing a nested list also raises during set membership.
- Root cause: `validate_config` checks that the outer section is a mapping, then coerces the names with `list(...)` and assumes its entries are hashable strings.
- Checked: The service accepts parsed config dicts directly. Config validation tests cover valid name lists, unknown names, bounds, and length mismatches. The scalar-section regression checks `initial_parameters` itself, with no malformed names inside a valid section.
- Red test: `assert not validate_config({"analysis_mode": "static_isotropic", "initial_parameters": {"parameter_names": 42, "values": [1]}}).ok`
- Confidence: high

## Non-blocking notes

- Static review of `bd0a549`. The proposed red tests and test suite were unrun.

## Verdict

BUGS_FOUND  
END_OF_REPORT