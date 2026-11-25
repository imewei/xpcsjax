# Codex review — shard ioviz
## Files read

xpcsjax/io/__init__.py — 20 — full  
xpcsjax/io/json_utils.py — 167 — full  
xpcsjax/io/nlsq_writers.py — 274 — full  
xpcsjax/utils/__init__.py — 32 — full  
xpcsjax/utils/color_limits.py — 32 — full  
xpcsjax/utils/logging.py — 1464 — full  
xpcsjax/utils/path_validation.py — 322 — full  
xpcsjax/viz/__init__.py — 62 — full  
xpcsjax/viz/datashader_backend.py — 387 — full  
xpcsjax/viz/diagnostics.py — 146 — full  
xpcsjax/viz/nlsq_plots.py — 1924 — full

## Bugs

### ioviz-B1: Homodyne reconstruction assigns fitted scaling to the wrong angles

- Location: xpcsjax/viz/nlsq_plots.py:561
- Severity: Required
- Failure scenario: A homodyne adapter fit receives angles `[90, 0]` and returns offsets `[1.0, 1.3]` in sorted-angle order. The 90° reconstruction uses `1.0`. The fitted NPZ, residual plots, and NRMSE then use the wrong fitted surface.
- Root cause: The adapter indexes scaling by `np.unique(phi)`. The reconstruction uses the raw data row index as the scaling index. Angle normalization and filtering can preserve unsorted data order.
- Checked: `NLSQAdapter._flatten_xpcs_data`, the adapter’s result assembly, `apply_angle_filtering_for_optimization`, and the post-fit and NRMSE callers. `test_adapter_flatten_phi_order.py` pins the sorted scaling index. The duplicate-angle visualization test exercises heterodyne, whose scaling contract differs.
- Red test:

  ```python
  def test_homodyne_reconstruction_uses_solver_phi_order():
      import numpy as np
      from types import SimpleNamespace
      from unittest.mock import Mock
      from xpcsjax.core.homodyne_model import HomodyneModel
      from xpcsjax.viz.nlsq_plots import _evaluate_c2_per_angle

      model = Mock(spec=HomodyneModel)
      model.parameter_names = ["D0", "alpha", "D_offset"]
      model.compute_c2_single_angle.side_effect = (
          lambda p, phi, c, o: np.full((2, 2), o)
      )
      result = SimpleNamespace(parameters=np.array([0.2, 0.4, 1.0, 1.3, 100., 0., 0.]))
      data = {"phi_angles_list": np.array([90., 0.])}
      c2 = _evaluate_c2_per_angle(model, result, data, {}, 90., phi_index=0)
      np.testing.assert_allclose(c2, 1.3)
  ```

- Confidence: high

### ioviz-B2: Homodyne reconstruction uses the requested wavevector instead of the fitted wavevector

- Location: xpcsjax/viz/nlsq_plots.py:576
- Severity: Required
- Failure scenario: The user requests `q=0.0054 Å⁻¹` and the loader selects the closest available bin at `0.0050 Å⁻¹`. The fit uses `0.0050`, while the reconstructed surface and its saved metadata use `0.0054`. This changes the reconstructed decay and the reported residuals and NRMSE.
- Root cause: The homodyne fit normalizes `wavevector_q_list[0]` into `data.q`. The `CombinedModel` visualization branch and artifact writer independently read the requested value from config.
- Checked: APS-U closest-bin selection, `_normalize_data_to_object`, `NLSQWrapper._create_residual_function`, `ConfigManager.get_model`, and the plot and NRMSE services. `test_laminar_flow_combined_model_does_not_raise` and the missing-config-q test do not exercise differing requested and selected values.
- Red test:

  ```python
  def test_reconstruction_uses_selected_q(
      monkeypatch, minimal_homodyne_config, converged_homodyne_result,
  ):
      import numpy as np
      from unittest.mock import Mock
      from xpcsjax.core.models import make_model
      from xpcsjax.viz.nlsq_plots import _evaluate_c2_per_angle

      cfg = minimal_homodyne_config
      model = make_model(cfg)
      compute = Mock(return_value=np.zeros((1, 2, 2)))
      monkeypatch.setattr(model, "compute_g2", compute)
      t = np.array([0., 0.1])
      data = {"t1": t, "t2": t, "wavevector_q_list": np.array([0.0050])}
      _evaluate_c2_per_angle(model, converged_homodyne_result, data, cfg, 0.)
      assert compute.call_args.args[4] == 0.0050
  ```

- Confidence: high

### ioviz-B3: The JSON size guard blocks saving fits with a modest diagnostic covariance

- Location: xpcsjax/io/json_utils.py:81
- Severity: Required
- Failure scenario: A diagnostics-enabled static fit with 15 independently scaled angles and three physics parameters produces a `33 × 33` diagnostic covariance. Saving the primary JSON or NPZ fails with `ValueError` because the 1,089-element diagnostic passes through JSON serialization even for NPZ output.
- Root cause: The wrapper attaches `diagnostic_covariance` to `nlsq_diagnostics`. Both persistence formats sanitize that block before writing. The encoder applies the same hard limit to small structured diagnostics and large correlation surfaces.
- Checked: `NLSQWrapper._post_process_results`, `_create_fit_result`, both `service.persist` writers, and the CLI and GUI persistence callers. `tests/test_io.py` pins rejection of oversized arrays. `tests/service/test_persist.py` covers ordinary result metadata without a covariance across the threshold.
- Red test:

  ```python
  def test_save_result_with_diagnostic_covariance(tmp_path):
      import numpy as np
      from xpcsjax.optimization.nlsq.results import OptimizationResult
      from xpcsjax.service.persist import save_results_npz

      result = OptimizationResult(
          parameters=np.ones(33), uncertainties=np.ones(33), covariance=np.eye(33),
          chi_squared=1., reduced_chi_squared=1., convergence_status="converged",
          iterations=1, execution_time=0.1, device_info={},
          nlsq_diagnostics={"diagnostic_covariance": np.eye(33)},
      )
      assert save_results_npz(result, tmp_path).is_file()
  ```

- Confidence: high

### ioviz-B4: A legal missing-uncertainty placeholder aborts plot artifact export

- Location: xpcsjax/viz/nlsq_plots.py:1885
- Severity: Required
- Failure scenario: A result carries `uncertainties=np.array(np.nan)`, a supported placeholder. The orchestrator renders any requested figures, then raises `IndexError` before writing either fitted artifact.
- Root cause: The visualization writer converts uncertainties to an array and immediately slices it as a vector. An absent or scalar placeholder produces a zero-dimensional array. An empty placeholder instead yields an empty uncertainty list beside nonempty parameter names and values.
- Checked: `OptimizationResult.__post_init__` admits these placeholders. `test_save_results_npz_replaces_offshape_uncertainties_with_nan` pins handling in the primary result writer. The visualization artifact tests use vector uncertainties.
- Red test:

  ```python
  def test_plot_artifacts_accept_scalar_uncertainty_placeholder(
      tmp_path, homodyne_model, converged_homodyne_result,
      synthetic_multi_angle_data, minimal_homodyne_config,
  ):
      import json
      import numpy as np
      from dataclasses import replace
      from xpcsjax.viz import generate_nlsq_plots

      result = replace(converged_homodyne_result, uncertainties=np.array(np.nan))
      generate_nlsq_plots(
          homodyne_model, result, synthetic_multi_angle_data, minimal_homodyne_config,
          tmp_path, plots=(), parallel=False,
      )
      path = tmp_path / "simulated_data" / "simulation_config_fitted.json"
      assert json.loads(path.read_text())["fit"]["parameters"]["uncertainties"] == [None] * 3
  ```

- Confidence: high

### ioviz-B5: Failure to open the optional log file can abort the CLI before fitting

- Location: xpcsjax/utils/logging.py:683
- Severity: Required
- Failure scenario: The configured log directory is writable, but the configured filename already names a directory. Log configuration raises `IsADirectoryError` before data loading. An existing log file that cannot be opened has the same failure mode.
- Root cause: The optional-file fallback catches `OSError` from creating the parent directory. The subsequent file-handler constructor opens the target outside that guard. Path validation accepts an existing directory with a filename component.
- Checked: `cli.commands.dispatch_command` applies file logging before dispatch, and the shipped laminar template enables it. `test_configure_handles_mkdir_failure` checks only the parent-directory failure. File-handler tests use writable targets.
- Red test:

  ```python
  def test_unopenable_log_target_keeps_console(tmp_path):
      from xpcsjax.utils.logging import configure_logging

      (tmp_path / "analysis.log").mkdir()
      config = {
          "console": {"enabled": True},
          "file": {"enabled": True, "path": str(tmp_path), "filename": "analysis.log"},
      }
      assert configure_logging(config) is None
  ```

- Confidence: high

### ioviz-B6: Nested log contexts overwrite inherited fields with `Ellipsis`

- Location: xpcsjax/utils/logging.py:1372
- Severity: Minor
- Failure scenario: `log_context(phase="fit")` inside `log_context(run_id="job")` replaces `"job"` with `Ellipsis`. Structured logs lose the run ID within the inner scope.
- Root cause: `log_context` forwards omitted keywords as `...`, while `set_log_context` recognizes only its private `_UNSET` object as omitted.
- Checked: `ContextFilter` copies the resulting value into records. `test_log_context_restores_on_exit` overrides the same field and checks restoration. It does not check inheritance of a field omitted by the inner scope. Confirmed with an in-memory probe.
- Red test:

  ```python
  def test_log_context_inherits_omitted_fields():
      import logging
      from xpcsjax.utils.logging import ContextFilter, log_context

      with log_context(run_id="job"), log_context(phase="fit"):
          record = logging.LogRecord("test", logging.INFO, __file__, 1, "message", (), None)
          ContextFilter().filter(record)
          assert (record.run_id, record.phase) == ("job", "fit")
  ```

- Confidence: high

### ioviz-B7: Removing managed log handlers leaves host logging disconnected

- Location: xpcsjax/utils/logging.py:725
- Severity: Minor
- Failure scenario: A library consumer installs a handler on the stdlib root logger, enables xpcsjax console logging, then reconfigures with both xpcsjax sinks disabled. Subsequent xpcsjax records stop reaching the host handler.
- Root cause: Configuration disables propagation when it installs managed handlers. Forced reconfiguration removes those handlers without restoring propagation when no handlers remain. The `PYTEST_CURRENT_TEST` branch ordinarily forces propagation on.
- Checked: `configure_from_dict` uses `force=True`. `test_configure_keeps_propagation_when_no_handler_installed` neither performs the transition nor disables the pytest override. Confirmed the stale propagation flag with an in-memory probe.
- Red test:

  ```python
  def test_removing_managed_sinks_restores_host_logging(monkeypatch):
      import logging
      from xpcsjax.utils.logging import MinimalLogger

      monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
      manager = MinimalLogger()
      manager.configure(console_level="INFO", force=True)
      manager.configure(force=True)
      assert logging.getLogger("xpcsjax").propagate
  ```

- Confidence: high

### ioviz-B8: Per-angle simulated heatmaps display the mean contrast and offset

- Location: xpcsjax/viz/nlsq_plots.py:1863
- Severity: Minor
- Failure scenario: Two angles are fitted with contrasts `0.2` and `0.6` and offsets `1.0` and `1.2`. Both heatmaps display `Contrast: 0.400` and `Offset: 1.100`, even though each surface uses its own fitted pair.
- Root cause: The rendering worker receives the scalar summary from `_unpack_result_params` for every angle. The surface evaluator separately resolves the actual angle-specific pair.
- Checked: `_evaluate_c2_per_angle`, `_render_one_angle_worker`, and the orchestrator fixtures. Existing tests exercise scalar fit metadata and angle-specific heterodyne evaluation. They do not compare the worker’s annotations with the pair used for that surface.
- Red test:

  ```python
  def test_simulated_plot_labels_use_angle_scaling(
      tmp_path, monkeypatch, homodyne_model, converged_homodyne_result,
      synthetic_multi_angle_data, minimal_homodyne_config,
  ):
      import numpy as np
      from dataclasses import replace
      from xpcsjax.viz import nlsq_plots as viz

      result = replace(
          converged_homodyne_result,
          parameters=np.r_[0.2, 0.6, 1., 1.2, converged_homodyne_result.parameters[-3:]],
          uncertainties=np.ones(7),
          covariance=np.eye(7),
      )
      data = dict(synthetic_multi_angle_data)
      for key in ("c2_exp", "phi_angles_list"):
          data[key] = data[key][:2]
      pairs = []
      monkeypatch.setattr(
          viz, "plot_simulated_data",
          lambda *args, **kw: pairs.append((kw["contrast"], kw["offset"])),
      )
      viz.generate_nlsq_plots(
          homodyne_model, result, data, minimal_homodyne_config, tmp_path,
          plots=("simulated",), parallel=False,
      )
      assert pairs == [(0.2, 1.), (0.6, 1.2)]
  ```

- Confidence: high

### ioviz-B9: Residual traces use unequal-time points when the axes have different origins

- Location: xpcsjax/viz/nlsq_plots.py:939
- Severity: Minor
- Failure scenario: For `t1=[0, 1, 2]` and `t2=[1, 2]`, the `t1 == t2` trace should sample cells `(1, 0)` and `(2, 1)` at times `1` and `2`. It instead samples `(0, 0)` and `(1, 1)` and labels them `0` and `1`.
- Root cause: `_resolve_extent` returns both coordinate vectors. The diagnostic discards the `t2` vector and takes the main array diagonal, which only represents equal times when the grids align by index.
- Checked: `plot_residual_map` documents separate `t` and `t2` axes and supports rectangular arrays. Both the worker and CLI comparison caller forward the two axes. The residual plot tests use a shared time vector.
- Red test:

  ```python
  def test_residual_trace_follows_equal_times():
      import matplotlib.pyplot as plt
      import numpy as np
      from xpcsjax.viz import plot_residual_map

      residual = np.arange(6.).reshape(3, 2)
      fig = plot_residual_map(
          1. + residual, np.ones((3, 2)), t=np.array([0., 1., 2.]), t2=np.array([1., 2.]),
      )
      try:
          line = fig.axes[2].lines[0]
          np.testing.assert_array_equal(line.get_xdata(), [1., 2.])
          np.testing.assert_array_equal(line.get_ydata(), [2., 5.])
      finally:
          plt.close(fig)
  ```

- Confidence: high

### ioviz-B10: Datashader crashes when a heatmap axis has one coordinate

- Location: xpcsjax/viz/datashader_backend.py:129
- Severity: Minor
- Failure scenario: A valid `3 × 1` raster with three y coordinates and one x coordinate raises `IndexError`. A comparison routed through Datashader fails before plot generation reaches artifact serialization.
- Root cause: Shape and finite-coordinate checks accept a singleton axis. Datashader 0.19.1 routes the regular grid through `QuadMeshRaster.is_upsample`, where bounds inference accesses the second coordinate.
- Checked: `generate_nlsq_plots` checks dimensional consistency without a minimum axis length. Its backend fallback covers missing Datashader, and the worker fallback retries the same renderer. `test_datashader_renderer_handles_rectangular_grid` exercises a `32 × 48` grid.
- Red test:

  ```python
  def test_datashader_handles_singleton_axis():
      import numpy as np
      from xpcsjax.viz.datashader_backend import DatashaderRenderer

      image = DatashaderRenderer(16, 16).rasterize_heatmap(
          np.full((3, 1), 1.2), np.array([0.]), np.arange(3.), vmin=1., vmax=1.5,
      )
      assert image.size == (16, 16)
  ```

- Confidence: high

## Non-blocking notes

- Possible duplicated code in `nlsq_plots.py`. Averaged heterodyne parameter and scaling extraction appears in two helpers. A shared extractor would keep their layout and scalar fallback rules together.
- Reviewed `bd0a549` with no shard edits. The test suite was skipped as requested. Two logging failures were confirmed in memory. The remaining findings were traced through source and existing tests, with red tests supplied for reproduction.

## Verdict

BUGS_FOUND  
END_OF_REPORT