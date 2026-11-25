# Codex review — shard data
## Files read
xpcsjax/data/__init__.py — 193 lines — full  
xpcsjax/data/angle_filtering.py — 449 lines — full  
xpcsjax/data/config.py — 549 lines — full  
xpcsjax/data/dataset.py — 82 lines — full  
xpcsjax/data/filtering_utils.py — 621 lines — full  
xpcsjax/data/hdf5_readers.py — 664 lines — full  
xpcsjax/data/memory_manager.py — 1032 lines — full  
xpcsjax/data/npz_cache.py — 529 lines — full  
xpcsjax/data/optimization.py — 564 lines — full  
xpcsjax/data/phi_filtering.py — 464 lines — full  
xpcsjax/data/preprocessing.py — 1064 lines — full  
xpcsjax/data/quality_controller.py — 2000 lines — full  
xpcsjax/data/types.py — 48 lines — full  
xpcsjax/data/validation.py — 733 lines — full  
xpcsjax/data/validators.py — 351 lines — full  
xpcsjax/data/xpcs_loader.py — 1972 lines — full  

## Bugs

### data-B1: Optimization angle filtering leaves the fitted q attached to an excluded angle
- Location: xpcsjax/data/angle_filtering.py:347
- Severity: Required
- Failure scenario: An APS-old dataset has rows `(phi=0°, q=0.0050)` and `(phi=90°, q=0.0052)`. Selecting only 90° leaves the first q at `0.0050`, which the homodyne fit extracts as its scalar q for the retained matrix.
- Root cause: The filter applies the selected indices to phi and C2 while retaining the original per-angle `wavevector_q_list`.
- Checked: `service/data.py::load_dataset` passes the filtered dictionary to the fitting path. `optimization/nlsq/core.py::_normalize_data_to_object` takes the first finite q. `tests/cli/test_data_pipeline_phi_filtering.py` uses identical q values for all angles.
- Red test: Call `apply_angle_filtering_for_optimization` with `phi_angles_list=np.array([0.,90.])`, `wavevector_q_list=np.array([.005,.0052])`, `c2_exp=np.ones((2,4,4))`, and an enabled range `[85,95]`. Assert `out["wavevector_q_list"].tolist() == [.0052]`.
- Confidence: high

### data-B2: Selective HDF5 loading silently broadcasts a short matrix into a full correlation surface
- Location: xpcsjax/data/hdf5_readers.py:649
- Severity: Required
- Failure scenario: Two selected APS-U bins have matrix shapes `(4,4)` and `(1,1)`. The second bin’s single value `1.5` becomes a 4×4 matrix with off-diagonal values of `3.0`. APS-old has the same assignment at line 326.
- Root cause: The reader derives the output shape from the first matrix, then assigns subsequent matrices into its buffer without checking that their shapes match. NumPy broadcasts the `(1,1)` result before the loader validates the assembled stack.
- Checked: `_load_from_hdf` validates the resulting stack. `test_aps_u_alloc_guard.py` checks probe dimensions, and `test_aps_u_selective_read_d7.py` uses uniformly sized bins.
- Red test: Use `_write_multi_bin_aps_u_file(path, n_q=1, n_phi=2)` from `test_aps_u_selective_read_d7.py`. Replace `c2_00002` in the correlation group with `np.array([[1.5]])`. Assert `_bare_loader()._load_from_hdf(str(path))` raises `XPCSDataFormatError`.
- Confidence: high

### data-B3: The first HDF5 bin’s dtype can truncate later float64 measurements
- Location: xpcsjax/data/hdf5_readers.py:632
- Severity: Required
- Failure scenario: A selected float32 first bin and float64 second bin produce a float32 stack. A second-bin off-diagonal value of `1.0000000001` returns as `1.0`, including after conversion to float64. The APS-old allocation at line 315 behaves the same way.
- Root cause: The selective path takes the batch dtype from the first bin. The earlier list-and-stack path promotes mixed floating dtypes when assembling the reconstructed matrices.
- Checked: `_convert_arrays_to_target_format` runs after the narrowing has occurred. The eager-reference comparison in `test_aps_u_selective_read_d7.py` uses only float64 datasets.
- Red test: In that test file, create two bins at one q. Recreate the first as `np.triu(np.full((4,4), 1.2, dtype=np.float32))` and the second as `np.triu(np.full((4,4), 1.0000000001, dtype=np.float64))`. Assert `_bare_loader()._load_from_hdf(str(path))["c2_exp"][1,0,1] == 1.0000000001`.
- Confidence: high

### data-B4: Integer-backed HDF5 matrices crash during reconstruction
- Location: xpcsjax/data/hdf5_readers.py:639
- Severity: Required
- Failure scenario: A square, integer-backed APS-U half-matrix raises NumPy’s output-casting error at in-place `/= 2`. The APS-old selective path and the shared quality-path reconstruction raise the same error.
- Root cause: Adding the half-matrix to its transpose keeps the integer dtype, then true division attempts to write float results into the integer array.
- Checked: The readers perform this operation before output dtype conversion. `core/diagonal_correction.py` explicitly accounts for integer-backed raw C2. The HDF5 selective-read fixtures use floats, and `test_preprocessing_diagonal_dtype.py` exercises a later processing stage.
- Red test: Use the one-bin APS-U fixture from `test_aps_u_selective_read_d7.py`. Recreate its matrix as `np.triu(np.full((4,4), 2, dtype=np.int32))`. Assert `_bare_loader()._load_from_hdf(str(path))["c2_exp"][0,0,1] == 2`.
- Confidence: high

### data-B5: The HDF5 readers and the quality filter read different configuration keys
- Location: xpcsjax/data/hdf5_readers.py:508
- Severity: Required
- Failure scenario: With `data_filtering.enabled: true` and `data_filtering.quality_threshold: 0.9`, both a low-quality bin and a clean bin survive. Conversely, setting the cutoff only inside `quality_filtering` activates matrix reads but leaves the quality mask disabled.
- Root cause: Both HDF5 readers gate passing matrices on `quality_filtering.enabled`. `XPCSDataFilter` and the data-loading schema read `quality_threshold` directly under `data_filtering`.
- Checked: `_get_selected_indices` forwards the configuration and matrix argument unchanged. `test_filtering_quality_score.py` tests scores directly. The selective-read quality parity test uses a nested zero cutoff, so both paths can pass it without applying a quality mask.
- Red test: Create two APS-U bins at one q and phi values `[0,90]`. Store `np.triu(np.full((4,4), .1))` and `np.triu(np.full((4,4), 1.2))`. Use `_bare_loader({"data_filtering": {"enabled": True, "quality_threshold": .9}})` and assert the returned phi list is `[90.]`.
- Confidence: high

### data-B6: The HDF5 metadata prefilter discards rows that satisfy the quality side of an OR filter
- Location: xpcsjax/data/hdf5_readers.py:519
- Severity: Required
- Failure scenario: At one q, the 0° bin matches the angle range and fails quality, while the 90° bin misses the angle range and passes quality. With `combine_criteria: OR`, both should survive. The reader returns only 0°.
- Root cause: The first pass applies the metadata mask without access to quality scores. It discards rows before the second pass can OR the quality mask with the metadata mask. APS-old uses the same two-pass design at line 212.
- Checked: `_combine_filter_masks` implements OR over the masks it receives. The final nonempty-selection guard accepts the surviving 0° row. The D7 tests cover empty metadata fallback and a nested zero cutoff.
- Red test: Use the two matrices from data-B5. Set `data_filtering` to `{"enabled": True, "quality_filtering": {"enabled": True}, "quality_threshold": .9, "combine_criteria": "OR", "phi_range": {"min": -5, "max": 5}}`. Assert that the APS-U reader returns phi `[0.,90.]`.
- Confidence: high

### data-B7: A fixed cache filename can silently serve the wrong frame window
- Location: xpcsjax/data/npz_cache.py:374
- Severity: Required
- Failure scenario: An unchanged four-frame HDF5 file is first loaded with frames 1–2 and `cache_filename_template: fixed.npz`. A second load requests frames 1–4 and returns the cached two-frame stack.
- Root cause: The writer records the applied start and end frames, but cache validation checks neither. The default template encodes the requested window, while a valid fixed template leaves every window at the same path.
- Checked: `load_experimental_data` validates the cache filename and opens a hit without reslicing. `test_cache_safety.py`, `test_cache_q_validation.py`, and `test_cache_source_file_validation.py` exercise other cache checks without changing the window under a fixed filename.
- Red test: With the four-frame APS-U fixture, load once through `load_xpcs_data` using `fixed.npz` and `end_frame=2`, then with the same config except `end_frame=4`. Assert the second result has `c2.shape[-2:] == (4,4)`.
- Confidence: high

### data-B8: A warm cache loses the record that filtering fell back to all rows
- Location: xpcsjax/data/xpcs_loader.py:878
- Severity: Required
- Failure scenario: A phi range matches no HDF5 bins with `fallback_on_empty=True`. The cold load returns the fallback data with `_degraded=True`. A fresh loader with the same config hits the cache and returns those same fallback rows with `_degraded=False`.
- Root cause: The loader saves the substituted arrays without the filtering degradation state. On a hit, it skips the filtering call that originally recorded the fallback.
- Checked: The NPZ writer stores the filter configuration hash, and the reader returns only the five arrays. `test_load_degradation_signal.py` checks the marker and final QC but does not round-trip a degraded filter selection.
- Red test: Use the one-bin APS-U fixture at 0°, enable data filtering with `phi_range={"min":85,"max":95}` and `fallback_on_empty=True`, then call `load_xpcs_data` twice with the same cache-enabled config. Assert `cold["_degraded"] and warm["_degraded"]`.
- Confidence: high

### data-B9: The loader ignores the shipped cache enable flag and cache directory
- Location: xpcsjax/data/xpcs_loader.py:797
- Severity: Required
- Failure scenario: Setting `experimental_data.use_caching: false` still reads or writes an NPZ cache. Setting `experimental_data.cache_directory` to a writable scratch directory still places the cache beside the HDF5 source, where a write failure can abort the load.
- Root cause: The loader reads `cache_file_path` for the directory and only `v2_features.cache_strategy` for enablement. The shipped templates use `cache_directory` and `use_caching`, and the configuration manager does not translate them.
- Checked: The four mode templates document both settings. `ConfigManager` resolves the directory path without remapping the field. `test_config_relative_data_paths.py` covers path handling, and `test_cache_source_file_validation.py` exercises the default cache policy.
- Red test: In `test_cache_source_file_validation.py`, use `_full_loader(tmp_path, "run.h5")` with a stubbed HDF read returning `_synthetic_data(1.5)`. With `use_caching=False`, assert `not list(tmp_path.glob("*.npz"))`. Then enable it and set `cache_directory=str(tmp_path/"scratch")`. Assert a cache exists under `scratch`.
- Confidence: high

### data-B10: The public loader rejects the modern `experimental_data.file_path` configuration
- Location: xpcsjax/data/xpcs_loader.py:677
- Severity: Required
- Failure scenario: A valid source file named only by `experimental_data.file_path` raises `XPCSConfigurationError` for missing `data_folder_path` before the loader opens the file.
- Root cause: The loader’s normalization handles the root-level legacy `data_file` shortcut and exits early for any `experimental_data` mapping. Its subsequent validation requires the legacy folder and filename keys.
- Checked: The interactive configuration generator emits `file_path` alone. `ConfigManager._normalize_experimental_data` can compose a path from legacy fields but cannot create those fields from `file_path`. The synthetic integration test supplies both legacy fields.
- Red test: In `test_loader_integration.py`, create `p = tmp_path / "source.npz"` with `_write_synthetic_npz_cache(p)`. Call `load_xpcs_data({"experimental_data":{"file_path":str(p)}, "analyzer_parameters":{"dt":1., "start_frame":1, "end_frame":6}})` and assert `data.c2.shape == (3,6,6)`.
- Confidence: high

### data-B11: The NPZ boundary accepts angle metadata with no corresponding matrix
- Location: xpcsjax/data/xpcs_loader.py:262
- Severity: Required
- Failure scenario: A source NPZ contains one 4×4 C2 matrix and two `(q, phi)` rows. Loading succeeds. A subsequent filter selecting only the second angle indexes beyond the C2 stack and raises `IndexError`.
- Root cause: The common I/O validator checks finiteness and the C2 stack’s rank and square dimensions. It never compares the leading C2 dimension with the lengths of the q and phi arrays.
- Checked: `npz_cache.load_from_cache` reaches this guard, and `apply_angle_filtering` indexes C2 by the selected phi indices. The data-loading guide specifies matching lengths. `test_loaded_array_validation.py::_good_data` itself uses mismatched counts even in its acceptance test.
- Red test: Save `_good_data(n_mat=1)` from `test_loaded_array_validation.py` to an NPZ and load it via `load_xpcs_data` with `dt=1`, `start_frame=1`, and `end_frame=4`. Assert loading raises `XPCSDataFormatError` before any angle filtering.
- Confidence: high

### data-B12: A full-turn phi range silently collapses to a single direction
- Location: xpcsjax/data/phi_filtering.py:41
- Severity: Required
- Failure scenario: Filtering `[0°,90°,180°]` with the valid range `[0°,360°]` returns only 0°. The presence of that one match prevents the fallback from restoring the other angles.
- Root cause: Independent endpoint normalization converts both 0° and 360° to 0° and discards the interval’s original width. The JAX and legacy paths share this helper. The optimization and plot wrappers normalize endpoints independently as well.
- Checked: `validate_target_ranges` permits both endpoints. The plot normalization test covers `[170°,190°]`, and the ±180° boundary tests cover normalization of individual angles.
- Red test: `assert filter_phi_angles([0.,90.,180.], target_ranges=[(0.,360.)])[0] == [0,1,2]`.
- Confidence: high

### data-B13: Default JAX output skips final QC symmetry and decay calculations
- Location: xpcsjax/data/quality_controller.py:871
- Severity: Required
- Failure scenario: At a QC pass threshold of 95, the same symmetric 4×4 dataset scores 96.25 and passes as NumPy, then scores 93.0 and fails as JAX. With QC enabled, the default loader output can therefore be marked degraded solely because of its array backend.
- Root cause: The loader converts to JAX before final QC. The advanced checks recognize only lists, tuples, and NumPy arrays, leaving symmetry and decay at their default values for JAX input.
- Checked: `_convert_arrays_to_target_format` defaults to JAX and `_comprehensive_quality_assessment` calls the advanced checks. `test_quality_controller_smoke.py` uses NumPy fixtures.
- Red test: In that smoke test, replace `_minimal_data()["c2_exp"]` with `jnp.asarray` of the same array. Run final QC with `pass_threshold=95`. Assert `result.metrics.symmetry_score == 100.0` and `result.passed`.
- Confidence: high

### data-B14: QC retains a hard validation error after successfully repairing the reported values
- Location: xpcsjax/data/quality_controller.py:1464
- Severity: Required
- Failure scenario: With `repair_nan_values=True`, a NumPy C2 matrix containing two repairable NaNs becomes entirely finite. Final QC still reports `Non-finite values found in c2_exp` and fails.
- Root cause: Post-repair checks update only basic metrics. The issue-removal heuristic looks for words such as `NaN` and `infinite` in the old issue, while the validator calls the defect `Non-finite values`.
- Checked: Final QC hard-fails on any remaining error. `test_repair_defaults_a3.py` checks repair mutation and logging without checking the final validation result.
- Red test: In `test_quality_controller_smoke.py`, set `d = _minimal_data(); d["c2_exp"][0,0,1] = d["c2_exp"][0,1,0] = np.nan`. Run final validation with `repair_nan_values=True`. Assert `np.isfinite(d["c2_exp"]).all()` and `result.passed`.
- Confidence: high

### data-B15: Incremental QC can reuse a passing result after the time axis becomes invalid
- Location: xpcsjax/data/quality_controller.py:1635
- Severity: Required
- Failure scenario: A controller passes final QC for a valid dataset. The caller changes `t1[0]` from 0 to −1 and repeats final QC with a previous-stage result. The cached pass is returned. A fresh controller rejects the same data.
- Root cause: The cache key includes stage, C2 shape, and a hash of C2 alone. Final validation also checks time, q, and phi, whose changes cannot invalidate that key.
- Checked: `validate_data_stage` returns a cache hit before the stage checks. The standard final-stage validator rejects negative times. The QC smoke tests use fresh controllers.
- Red test: With `_minimal_data()` and one default controller, save `raw = qc.validate_data_stage(d, QualityControlStage.RAW_DATA)`. Run final validation with `raw`, set `d["t1"][0] = -1`, and assert `not qc.validate_data_stage(d, QualityControlStage.FINAL_DATA, raw).passed`.
- Confidence: high

### data-B16: A skipped preprocessing stage can leave the loaded dataset marked clean
- Location: xpcsjax/data/xpcs_loader.py:1769
- Severity: Required
- Failure scenario: Enable Savitzky–Golay denoising with `window_length=3` and `polyorder=4` on a four-frame dataset. SciPy rejects the parameters. The pipeline continues, returns the undenoised data, and the loader returns `_degraded=False`.
- Root cause: The pipeline deliberately permits partial success with `abort_on_error=False` and records the failed stage. The loader’s success branch ignores failed `stage_results` and provenance errors when setting its degradation marker.
- Checked: `PreprocessingPipeline.process` records the failure before continuing. `test_reduce_noise_dtype.py` covers valid denoising, and `test_load_degradation_signal.py` checks final QC without exercising partial preprocessing success.
- Red test: In the NPZ round-trip test, enable preprocessing with `stages={"reduce_noise":{"enabled":True,"method":"savgol","window_length":3,"polyorder":4}}`. Assert `load_xpcs_data(config_dict)["_degraded"]`.
- Confidence: high

### data-B17: The schema validator raises on a field type it has already identified as invalid
- Location: xpcsjax/data/config.py:345
- Severity: Minor
- Failure scenario: A YAML value `dt: "bad"` reaches `validate_config_schema`. It records a wrong-type error, then raises `TypeError` from `math.isfinite` instead of returning an invalid `ConfigValidationResult`.
- Root cause: Semantic validation reruns against raw values after the type check and calls numeric validators without filtering out fields that failed type validation. The small-`dt` warning path also assumes a number.
- Checked: YAML safely parses the quoted value as a string. The existing null and malformed section tests in `test_debug_audit_2026_08_05.py` do not cover malformed scalar values inside a valid mapping.
- Red test: `result = validate_config_schema({"experimental_data":{"data_folder_path":".","data_file_name":"CLAUDE.md"},"analyzer_parameters":{"start_frame":1,"end_frame":4,"dt":"bad"}})`. Assert `not result.is_valid` and `any("dt" in error for error in result.errors)`.
- Confidence: high

### data-B18: Memory-manager shutdown overwrites process GC thresholds it never changed
- Location: xpcsjax/data/memory_manager.py:996
- Severity: Minor
- Failure scenario: A host application creates a manager with monitoring and GC optimization disabled, then changes its own GC thresholds. Closing the manager restores its constructor-time snapshot and undoes the host’s setting.
- Root cause: The manager snapshots process-wide thresholds at construction and unconditionally restores them at shutdown, without tracking whether it changed them.
- Checked: `XPCSDataLoader.close` invokes this shutdown, and explicit shutdown is also part of the manager API. `test_memory_manager_logging.py` covers shutdown and collection but does not assert threshold ownership.
- Red test: In `test_memory_manager_logging.py`, monkeypatch `gc.get_threshold` to return `(700,10,10)` and `gc.set_threshold` to record writes. Construct `AdvancedMemoryManager({"memory":{"enable_monitoring":False,"gc_optimization":False}})`, call `shutdown()`, and assert that no threshold write was recorded.
- Confidence: high

## Non-blocking notes

- Possible Duplicated Code. HDF5 half-matrix reconstruction appears in the two selective readers and the loader’s shared helper. A common reconstruction routine would give both formats one place to enforce shape and dtype behavior.
- Selected failures were checked with read-only, in-memory probes. The pytest suite was not run.

## Verdict
BUGS_FOUND  
END_OF_REPORT