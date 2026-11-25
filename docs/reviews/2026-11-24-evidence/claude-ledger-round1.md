# Round 1 ledger (Codex claims -> Claude verification)

Verdicts: CONFIRMED (red test in tests/review_r1/), REJECTED, DESIGN (behaviour as claimed; maintainer decision, not fixed).

## optrest (6 claims: 4 confirmed, 2 design)
- B1 CONFIRMED parallel_accumulator.py:831 shutdown(wait=True) after timeout -> fix wait=False (sibling fn at :202-211 already does)
- B2 CONFIRMED recovery.py:316 "converged_with_recovery" not in ConvergenceStatus -> emit "converged"
- B3 DESIGN wrapper_out_of_core_route.py:139 DOF n_physical under constant; pinned by test_laminar_mode_constant_token_is_not_pinned + per_angle_mode.py:119-126 contract; ~1e-5 relative
- B4 DESIGN recovery.py:198/228 overrides x_scale; verbatim upstream port (98244ac); honouring it moves rtol=1e-10 goldens
- B5 CONFIRMED memory.py:373 default 0.75 masks NLSQ_MEMORY_FRACTION -> default None. heterodyne_memory.py:231 same default, untested (callers nlsq/__init__.py:873, heterodyne_stratified_ls.py:861)
- B6 CONFIRMED minor parameter_utils.py:630,641 isclose(atol=0.1) pools bins -> == phi

## config (11 claims: 7 confirmed, 3 design, 1 rejected)
- B1 CONFIRMED manager.py:1475 legacy pair overwrites explicit file_path (template :102 promises file_path wins; cli/config_template.py:153 injects file_path) -> derive legacy pair from file_path when truthy; touches tests/config/test_config_relative_data_paths.py
- B2 DESIGN equal bounds kept in optimizer vector is stated convention (tests/optimization/test_fixed_parameters_integration.py:11-18, data_prep.py:224); changing vary changes DOF
- B3 CONFIRMED heterodyne_parameter_space.py:509,548 typos silently narrow/skip; configuration.rst:290 promises ValueError -> raise
- B4 CONFIRMED heterodyne_parameter_space.py:399,887 bool("false") True -> local parse/reject helper (config must stay JAX-free; no safe_bool import from optimization/)
- B5 DESIGN stated root cause wrong; midpoint behaviour documented manager.py:787-791; preview-vs-fit source is a maintainer call
- B6 CONFIRMED heterodyne_parameter_space.py:296,325 .get(k, {}) keeps YAML null -> `or {}`
- B7 CONFIRMED parameter_space.py:179 float(None) -> configured branch only when both non-None
- B8 CONFIRMED manager.py:234 load_config doesn't clear _cached_param_manager -> clear at top
- B9 DESIGN null values -> midpoints documented manager.py:782-791; no consumer reads contrast_N from accessor
- B10 CONFIRMED minor heterodyne_parameter_space.py:199-204 partial write before raise -> validate all keys first
- B11 REJECTED validate_correlation_inputs has no production caller, not exported; [0, inf] satisfies documented "strictly increasing"
(B7/B8/B10 confirmed under "public method on exported class counts as reachable")

## core (10 claims: 7 confirmed, 2 design, 1 rejected)
- B1 CONFIRMED heterodyne_model_stateful.py:234 sync_time_axis trims LEADING points on shrink; loader clamps end_frame with only a warning (xpcs_loader.py:1452-1455) -> t=2.1..10.0 instead of 0.1..8.0. Proposed fix UNVERIFIED (write clamped end_frame back to analyzer_config); changing sync_time_axis shifts every short-data fit. Needs own root-cause pass before fixing.
- B2 DESIGN physics_nlsq.py:140 64 substeps deliberate (comment :131-139), pinned at 1e-3 by tests/core/test_physics_nlsq_degenerate_grid.py; changes stratified numerics
- B3 CONFIRMED adapter.py:605,1012 compute_g1_batch without time_grid -> 10001-pt fallback + clip (API-only use_adapter=True, times > 10000*dt) -> thread time_grid
- B4 REJECTED jax_backend.py:132 cache-key collision needs a non-uniform axis; loader emits uniform linspace (xpcs_loader.py:1513) and matrix mode assumes uniform dt (jax_backend.py:403-417)
- B5 CONFIRMED jax_backend.py:1269 NaN data over infinite sigma stays NaN; docstring :1241 promises zero -> masked where
- B6 CONFIRMED homodyne_model.py:352 static templates carry analysis_settings.model_description without legacy flags -> resolves laminar_flow (7 params) -> legacy branch only when static_mode/isotropic_mode present
- B7 CONFIRMED heterodyne_jax_backend.py:361 (4,1) broadcasts against (4,4); no shape guard in HeterodyneModel.compute_residuals -> raise ValueError in heterodyne_model_stateful.py before :358
- B8 CONFIRMED diagonal_correction.py:269 float scatter into int array truncates (1 vs 1.5) -> promote to float in both wrappers (:274, :285) + vmapped batch path
- B9 DESIGN heterodyne_models.py:296 hard-clip vs kernel smooth-clip; differ only near bounds, no consumer
- B10 CONFIRMED physics.py:203-213 2-D input -> valid=True with 0 checked -> invalid when ndim != 1

## optescape (8 claims: 6 confirmed, 1 design, 1 rejected)
- B1 CONFIRMED-BUT-NUMERIC hierarchical.py:437,497-511 convergence compares physics before Stage 1 vs after Stage 2 (which never touches them) -> one-sweep "convergence" typical. Fix adds sweeps => individual-mode L2 numerics change. LEAD: likely reclassify DESIGN unless fix can be limited to the success label.
- B2 CONFIRMED-RISKY cmaes_wrapper.py:703-733 hardcodes success=True, ier=1 for refinement even when max_nfev exhausted. Fix must separate "converged" from "adoptable" (gates cmaes_wrapper.py:1185, core.py:2600, :1267, core.py:2634-2670). Success-gate area has regressed 3x before (see memory) — own root-cause pass.
- B3 CONFIRMED cmaes_wrapper.py:677-711 TRF-only kwargs passed with method="hybrid_streaming" -> TypeError swallowed, refinement silently skipped (config.py:754 accepts streaming; no template sets it)
- B4 CONFIRMED cmaes_wrapper.py:995 explicit max_generations overridden (3 -> 300); docstring :266 says int overrides -> guard with `is None`; keep line text (test_cmaes_memory_sizing_order.py:61 indexes it)
- B5 DESIGN multistart workers alive after 1800 s timeout fallback; documented as accepted multistart.py:1372-1381,1420-1424; hang at exit not proven
- B6 CONFIRMED cosmetic multistart.py:1187 overwrites full-length initial_params (len 8 vs 9) -> delete line
- B7 REJECTED adaptive_regularization.py:397 zero gradient real but method has no callers (dead code)
- B8 CONFIRMED minor config.py:749 valid list omits "averaged" -> false warning at :617 -> add it (+ comment :300)

## gui (gui + service; 14 claims: 12 confirmed, 1 design, 1 rejected)
- B1 CONFIRMED service/persist.py:247 labels params with config active-list order; solver tail is canonical (core.py:475, 2460) -> mislabelled values ([alpha, D0, D_offset]) -> names from registry order; touches fake cms in tests/service/test_persist.py:169-285
- B2 DESIGN service/data.py:92 q unsliced on phi filter; angle_filtering.py:231 documents q "preserved unchanged", fit uses q[0] (core.py:788)
- B3 CONFIRMED service/data.py:77 `diffs[j] > tol` False for NaN -> index 0 taken (reachable `--phi nan`) -> `if not (diffs[j] <= tol)`
- B4 CONFIRMED viz/nlsq_plots.py:576 q from config while fit uses data q (core.py:788) -> prefer data["wavevector_q_list"][0]; changes plots (postfit.py:121, nlsq_plots.py:1791)
- B5 CONFIRMED gui/project/persist.py:117 restores "running" -> run_controller.py:59 blocks Run -> map queued/starting/running to KILLED on load
- B6 CONFIRMED narrow gui worker handle.py:69 get() blocks on partial frame (parent keeps write end) -> close self._queue._writer after start (private attr); FIX UNTESTED
- B7 CONFIRMED main_window.py:645 _reset_run_view_state leaves _pending_result_dir set (result_presenter.py:163)
- B8 CONFIRMED config_dialogs.py:295 non-atomic write_text -> temp + os.replace (as gui/project/persist.py:78)
- B9 CONFIRMED service/persist.py:334 fixed ".tmp" name races concurrent workers (io/nlsq_writers.py:29-33 documents it) -> mkstemp
- B10 CONFIRMED service/persist.py:606 NamedTemporaryFile outside try; callers unguarded (worker.py:76, commands.py:190)
- B11 CONFIRMED gui/project/persist.py:103 non-dict payload -> AttributeError; handler catches only ValueError/OSError (project_dialog_handler.py:165)
- B12 CONFIRMED main_window.py:619 None summary never sets flag (result_loader.py:57) -> set when summary None and status DONE
- B13 CONFIRMED data_inspect.py:29 tuple(None) TypeError uncaught by dialog -> tuple(obj.shape or ())
- B14 REJECTED service/config.py:190,194 validate_config has no in-tree caller; tests/gui/test_gui_debug_fixes.py:22 suggests non-raising intended
- caveats: B1 not exercised through a real fit (order read from core.py) — confirm end to end before fixing; B4 fix changes plotted surfaces when config q != measured q

## cli (cli + runtime + device + __init__; 10 claims: 9 confirmed, 1 design)
- B1 CONFIRMED = DUPLICATE of config-B1 (same root, one fix at config/manager.py:1456; also fixes config_template.py:357)
- B2 CONFIRMED cli/commands.py:188-193 merges plots whenever plotting ran; json-only save writes no NPZ (service/persist.py:478) so a stale NPZ passes freshness guard (persist.py:570) -> gate merge on output_format in ("npz","both"); gui/ipc/worker.py:76 has same ungated merge
- B7 CONFIRMED commands.py:189 second resolve_plots_dir outside try -> FileExistsError (plot_backend.py:69) -> same site as B2, use output_dir/"plots" directly
- B3 CONFIRMED cli/main.py:71 --intra_op_parallelism_threads in XLA_FLAGS aborts XLA ("Unknown flag", jaxlib 0.11.1) -> drop at main.py:71, cli/xla_config.py:62,:174
- B4 CONFIRMED plot_families/simulated.py:113-114 reads only analyzer_parameters.scattering; model falls back to top-level scattering (heterodyne_model_stateful.py:130) -> add fallback
- B5 CONFIRMED simulated.py:148 KeyError 'D_offset' (manager.py:899 drops fixed params) swallowed -> zero plots -> merge fixed_parameters into init_params
- B6 CONFIRMED config_template.py:238 discards parsed value; empty file / scalar report VALID -> ok=False unless dict; same pattern config_handling.py:117
- B8 CONFIRMED minor config_template.py:149 splitlines()[0] truncates newline-containing scalar -> unparseable YAML -> json.dumps string
- B9 DESIGN xla_config.bash:71-73 keeps existing device-count token (mirrors xpcsjax/__init__.py:87-94); reviewer's red test was itself broken
- B10 CONFIRMED cli/__init__.py:31 `from xpcsjax.cli import main` returns the module not the function -> drop from _IMPORTS/__all__ (changes __all__; maintainer-visible)
- caveat B5: subset active_parameters not checked — merging fixed dict may be a partial fix

## optwrap (wrapper.py, wrapper_stratified_route.py, core.py, adapter.py; 10 claims: 8 confirmed, 1 design, 1 rejected)
Verifier: none of the 8 fixes changes golden-path numbers (goldens use ascending phi, no fixed params, wrapper route, default memory fraction).
- B1 CONFIRMED wrapper.py:3588 theory on sorted np.unique(phi), :3778 indexed by raw row -> gather via searchsorted(phi_unique, phi); identity for ascending phi
- B2 CONFIRMED wrapper.py:2340-2347 (+ :3111/3141) raw-order sigma grid indexed with sorted-phi indices (:2637 documents sigma follows phi_raw) -> map through phi_raw
- B9 CONFIRMED core.py:2333 raw row indices on frozen scaling returned sorted (parameter_utils.py:809) -> searchsorted(unique(phi), phi)
  (B1/B2/B9 ONE ROOT: sorted-unique phi vs raw row order. Fix at each site with identity-for-ascending gather; do NOT sort at entry — that reorders returned per-angle params.)
- B3 CONFIRMED wrapper.py:3218 IndexError: bounds from get_all_parameter_names() (:2810) drop inactive names (8) vs dense labels (13); sequential path only
- B4 DESIGN wrapper.py:62-65 docstring says covariance s² uses expanded DOF; only diagnostics covariance (:2032-2046) does; rescaling changes uncertainties ~1% on every default auto->averaged fit
- B5 CONFIRMED wrapper.py:1759 DOF counts restored fixed slots (wrapper_stratified_route.py:401-438 already applies "fixed must not consume a DOF"); other listed sites (wrapper.py:3211, wrapper_stratified_route.py:578, core.py:2906, adapter.py:1234) NOT run by verifier
- B6 REJECTED sigma=None unreachable: core.py:759-767 raises, :814 defaults it; only a test stub reaches it
- B7 CONFIRMED adapter.py:1116 `len(t1) % n_angles == 0` fires for shared 1D marginals -> adapter fails, falls back -> guard with len(t1) == n_points
- B8 CONFIRMED adapter.py:611,1018 pass L=1.0 (gap multiplies shear prefactor, jax_backend.py:461): 0.87203 vs 0.85456 -> read L from data, add L to ModelCacheKey (adapter.py:221); adapter is opt-in
- B10 CONFIRMED wrapper.py:871,:1621, wrapper_stratified_route.py:215 never pass configured nlsq.memory_fraction (:304-320 only logs it) -> pass when key present. Related to optrest-B5 (same selector default) — fix together.

## ioviz (io + utils + viz; 10 claims: 6 confirmed, 2 design, 2 rejected)
- B1 CONFIRMED viz/nlsq_plots.py:561 indexes per-angle scaling by data row while the fit indexes by sorted np.unique(phi) (wrapper.py:3588,3685; adapter.py:1085); angle_filtering.py:279-295 keeps row order -> homodyne branch: searchsorted(np.unique(phi_list), phi_list[i]); heterodyne branch stays row-indexed. SAME FAMILY as optwrap B1/B2/B9 (sorted-unique phi vs row order). Callers nlsq_plots.py:1791, service/fit_quality.py:137, cli/plot_families/postfit.py:121.
- B2 CONFIRMED = DUPLICATE of gui-B4 (nlsq_plots.py:576, also :1685: config q vs fit's data q) -> prefer finite data["wavevector_q_list"][0]; keep ValueError when both missing; shifts persisted NRMSE when they differ
- B3 CONFIRMED service/persist.py:136,:401 push n×n diagnostic_covariance (wrapper.py:2045, when diagnostics.enabled) through json_safe limit 1000 -> save fails at 32+ solver params, NPZ too -> drop from JSON copy, store as NPZ array; do NOT raise the limit (tests/test_io.py:94,114)
- B4 CONFIRMED public-API-only nlsq_plots.py:1885-1898 slices 0-d uncertainties -> IndexError swallowed by service/plots.py; no in-tree fit path emits 0-d -> ravel + NaN fill. Low priority.
- B5 CONFIRMED utils/logging.py:680-689 file handler opened outside OSError guard (:648); cli/commands.py:67 unguarded -> CLI aborts before fit on unwritable log dir -> try/except OSError, console-only
- B6 CONFIRMED latent logging.py:1361-1364 log_context defaults `...` but set_log_context skips only _UNSET (:1344) -> run_id becomes Ellipsis; exported, no in-package caller -> default to _UNSET
- B7 DESIGN logging.py:723-729 propagate not restored; restoring collides with pinned contract tests/test_logging_quality_gate.py:160-183
- B8 DESIGN scalar mean in annotation panel is documented summary contract (nlsq_plots.py:309, :356-374); label-only
- B9 REJECTED index diagonal intended (nlsq_plots.py:939 comment); loaders always return t2 = t1.copy() (hdf5_readers.py:659-660)
- B10 REJECTED datashader crash needs a length-1 time axis; loaders give square matrices on a shared axis, 1 frame has nothing to fit
- caveat: no end-to-end fit run for B1-B3 (code trace + adapter index probe)

## data (18 claims: 15 confirmed, 3 rejected) — verifier rule: CONFIRMED needs an in-package caller
- B1 CONFIRMED?? angle_filtering.py:347-349 slices phi + c2 but not wavevector_q_list; fit takes q_list[0] (core.py:788,:1963); bites where q differs per row (APS-old 10% tolerance, NPZ). CONFLICT with gui-B2 (DESIGN: same omission in service/data.py; angle_filtering.py:231 documents q "preserved unchanged"). LEAD MUST ADJUDICATE both together.
- B2 CONFIRMED hdf5_readers.py:649 (APS-U), :326 (APS-old) a (1,1) bin broadcasts to n_t×n_t -> raise XPCSDataFormatError on shape mismatch
- B3 CONFIRMED malformed-file-only :632,:315 dtype from first bin narrows a float64 later bin -> same guard (np.can_cast safe)
- B4 CONFIRMED :639,:318, xpcs_loader.py:1074 in-place /= 2 on int32 raises UFuncOutputCastingError -> allocate np.result_type(dtype, float32) (float files bit-identical)
- B5 CONFIRMED readers gate on quality_filtering.enabled (hdf5_readers.py:508,:172) but filter reads flat quality_threshold (filtering_utils.py:338) -> only both keys together work -> read both
- B6 CONFIRMED hdf5_readers.py:519-531,:212-222 metadata pre-filter drops rows before quality can OR them back -> skip pre-filter when combine_criteria == "OR"
- B7 CONFIRMED npz_cache.py:298-304 records frames, validate_cache_q_vector (:374) never compares -> serves 2-frame stack for 4-frame request -> store start/end, CacheStaleError on mismatch (warn-only when key absent)
- B8 CONFIRMED xpcs_loader.py:878 cold load _degraded=True, warm cache hit False -> skip _save_to_cache when load_degradations non-empty
- B9 CONFIRMED (use_caching half) xpcs_loader.py:797,:847,:877 ignore use_caching (all templates ship it) -> gate on exp_config.get("use_caching", True). cache_directory half = DESIGN (would move every template user's cache)
- B10 CONFIRMED xpcs_loader.py:537 early return, :677 requires legacy keys; config_template.py:358 emits file_path only -> split file_path when legacy keys missing. OVERLAPS config-B1/cli-B1 — decide precedence once, fix coherently.
- B11 CONFIRMED xpcs_loader.py:262-268 never compares c2.shape[0] with phi length -> add check; BREAKS tests/data/test_loaded_array_validation.py::_good_data (3 phi, 2 matrices) -> fixture needs fixing; leave q out
- B12 CONFIRMED edge phi_filtering.py:41-46 (0,360) normalises to (0,0) -> keeps only 0° -> if max-min >= 360 return (-180,180); same at angle_filtering.py:314-315,:413-414
- B13 CONFIRMED (needs quality_control.enabled, off by default) quality_controller.py:871 isinstance gate excludes JAX arrays -> score differs 96.25 vs 93.0 -> np.asarray first
- B14 REJECTED stale issue survives repair (quality_controller.py:1490) but _validate_loaded_arrays (xpcs_loader.py:240-248) rejects NaN c2 before QC on both load paths
- B15 REJECTED fingerprint hashes c2 only (:552) but loader builds a fresh controller per load (xpcs_loader.py:1697)
- B16 CONFIRMED xpcs_loader.py:1769 ignores stage_results (preprocessing.py:364-374 warns stage skipped) -> _record_degradation per failed stage
- B17 REJECTED validate_config_schema TypeError real (config.py:345) but no in-package caller
- B18 CONFIRMED minor memory_manager.py:996 restores unconditionally (XPCSDataLoader.close calls it) -> flag-gated restore
- caveat: APS-old variants of B2-B6 read, not run
## opthet (heterodyne_*.py; 13 claims: 9 confirmed, 4 design, 0 rejected)
Verifier: none of the 9 fixes changes default-path numerics (weights=None, trf, regularization off, all 14 physics varying, sorted phi, no escapes).
- B1 CONFIRMED heterodyne_engine_route.py:581 unsorted phi ([75,0]): engine optimum SSR 4.9e-31 but reported chi_squared 8.8 -> argsort phi/c2 at entry, un-permute scaling head/cov/per-angle diagnostics on return (averaged already correct). SORTED-vs-RAW PHI FAMILY.
- B2 CONFIRMED heterodyne_stratified_ls.py:1404-1409 offsets returned under wrong angle labels (solve right, labelling wrong) -> un-permute with inverse argsort before build_hybrid_streaming_result. SAME FAMILY.
- B3 CONFIRMED heterodyne_stratified_data.py:170 zero-weight pixel contributes residual -1.0 (in-memory kernel zeros it: heterodyne_jax_backend.py:361,470) -> np.where(w>0, 1/sqrt(w), inf); fix docstring :113; hybrid streaming also reads sigma (:346)
- B4 CONFIRMED heterodyne_core.py:3399-3404 penalty-row Jacobian NaN at uniform seed (std at zero variance) -> individual+adaptive-regularization fit fails with 0 nfev -> differentiable std; same pattern heterodyne_stratified_ls.py:394. (Same class as the laminar CV² NaN fixed in 621e647.)
- B5 CONFIRMED heterodyne_core.py:2023 with a fixed param, escape gets expanded 14-vector vs 13 optimizer coords -> broadcast error, CMA-ES never runs -> take x_warm in optimizer coords; same at multistart :2300-2315
- B6 DESIGN heterodyne_adapter.py:540-551 fit_jax reports success after max_nfev; honouring backend verdict re-fits via NLSQWrapper -> changes sequential-path numerics
- B7 CONFIRMED heterodyne_engine_route.py:555 method="lm" + bounds returns failed result, fallback never runs -> coerce lm->trf as heterodyne_core.py:1496,3532 and heterodyne_constant_mode.py:248 do
- B8 DESIGN heterodyne_stratified_ls.py:806-832 raise on f1=0 setup deliberately mirrors laminar (strategies/stratified_ls.py:534-559)
- B9 CONFIRMED heterodyne_core.py:1674,1783 covariance DOF ratio 314/316 (penalty rows miscounted) -> rescale_covariance_dof with correct rows; reported covariance only; twin site :3931
- B10 CONFIRMED heterodyne_config.py:1028 regularization `enable: false` ignored (nested mode still applied) -> skip nested mode when disabled; update comment heterodyne_stratified_ls.py:942; test_heterodyne_config.py:162 must still pass
- B11 DESIGN heterodyne_engine_route.py:419-430 uniform weights dropped; fix direction open, each changes weighted-fit objective; "argmin unchanged" comment false under soft_l1
- B12 CONFIRMED heterodyne_result_builder.py:765 exact-match on legacy max-nfev string misses NLSQAdapter's real message ("The maximum number of function evaluations is exceeded.") -> result graded failed/poor -> also match case-insensitively as wrapper.py:3991 does; labels only; keep legacy string (test_heterodyne_hybrid_streaming.py:231)
- B13 DESIGN heterodyne_core.py:608,720 sequential path subtracts n_physics + 2*n_phi; reporting only, pinned by test_heterodyne_individual_joint.py:142

## optstrat (strategies/; 16 claims: 10 confirmed, 4 design, 2 rejected)
Verifier: no confirmed fix changes the golden-covered homodyne path (ascending phi, unit sigma, standard solve); B4/B5/B8 change reported covariance only, never popt/chi².
- B1 CONFIRMED stratified_ls.py:100-102,155 forwards raw-order sigma and drops phi_raw; engines gather by sorted index (residual_jit.py:503, residual.py:635) -> permute sigma axis 0 to sorted order in create_stratified_chunks (as hybrid_streaming.py:1071-1090); no-op for ascending phi / sigma=None; heterodyne callers (heterodyne_engine_route.py:481, het stratified-LS) carry no phi_raw -> must add. SORTED-vs-RAW PHI FAMILY.
- B2 CONFIRMED heterodyne_hybrid_streaming.py:351-354 sigma_3d gathered with sorted index on input-order sigma (heterodyne_stratified_data.py:162-170) -> map sorted index to input row. SAME FAMILY.
- B3 CONFIRMED minor stratified_ls.py:551-597 preflight perturbs only D0; zero derivative at contrast 0 -> raises (contrast lower bound 0.0: parameter_registry.py:230; wrapper only clips :3435) -> perturb every free param
- B4 CONFIRMED stratified_ls.py:883,938 n_real_points includes masked t1==t2 rows (covariance.py:5-9 says not observations) -> subtract diagonal/zero-sigma rows at :883; changes >=1M laminar stratified-LS uncertainties slightly (COVARIANCE CHANGE — lead to weigh vs DESIGN)
- B5 CONFIRMED out_of_core.py:513 count includes diagonal rows the kernel zeroes (parallel_accumulator.py:366) and DOF ignores fixed params (unlike stratified_ls.py:873): got 8/11, want 1 -> count t1!=t2 rows, subtract fixed count (COVARIANCE CHANGE)
- B6 CONFIRMED sequential.py:780-806 uniform / n_points weighting averages slots an angle never fitted -> NaN covariance -> per-parameter weights × isfinite(variance); default inverse_variance untouched
- B7 DESIGN sequential.py:633-645 pinv documented as acceptable for per-angle covariance
- B8 CONFIRMED heterodyne_hybrid_streaming.py:926-937 accepts indefinite Hessian (checks only finiteness + diagonal sign); hybrid_streaming.py:1373 same gap (by reading) -> eigvalsh min > 0 at both; keep "singular" in existing log (two tests assert on it) (COVARIANCE CHANGE)
- B9 DESIGN hybrid_streaming_layers.py:343,377 ignores regularization.enable; pinned "ignored by design" (test_heterodyne_stratified_anti_degeneracy_parity.py:176-183, test_laminar_execute_layers.py:131); honouring changes default laminar streaming numerics
- B10 CONFIRMED fallback_chain.py:337 passes ConfigManager; hybrid_streaming.py:155-156 reads .hybrid_normalize -> AttributeError -> silent fall to CHUNKED -> use hybrid_* branch only for NLSQConfig. FIX TOGETHER WITH B11.
- B11 CONFIRMED hybrid_streaming.py:227-231 returns ydata - residual_fn(x) but wrapper passes a model (wrapper.py:1210,1386) -> broadcast error. Fixing B10 alone would run a silently wrong objective.
- B12 REJECTED executors.py:403 broadcast error real, but StreamingExecutor/get_executor have no in-package caller and no docs mention (export-only); full-batch contract pinned by tests/optimization/test_strategy_executors.py:225. Dead export.
- B13 DESIGN out_of_core.py:290 reads optimization.max_iterations not optimization.nlsq.max_iterations; using the nlsq key lifts OOC cap from 50 full-data passes to 10000 (xpcsjax_laminar_flow.yaml:324); keys count different things
- B14 CONFIRMED minor out_of_core.py:625-693 exactly-zero gradient never passes chi2_new < chi2_ref -> "line_search_stalled", success False -> treat max|active_Jtr| == 0 as converged before damping loop (~:577); popt unchanged
- B15 CONFIRMED minor sequential.py:931 full-length x_scale re-validated against free count and dropped with a warning -> slice array-valued x_scale/diff_step/f_scale by free_mask; only sequential + fixed param + numeric x_scale
- B16 REJECTED executors.py:173 bounds=None TypeError — same dead export-only executor code
(count corrected by verifier: 11 confirmed, 3 design, 2 rejected)

## LEAD ADJUDICATIONS (one rule applied to all shards)
Rule: CONFIRMED = defect reachable from an in-package caller or the documented public surface (lazy exports / __all__, CLI flags, documented config keys), AND the fix does not change fit numbers (popt / chi²) on a default path. Otherwise DESIGN (behaviour as claimed, maintainer decision) or REJECTED (not a defect / unreachable).
- data-B1 -> DESIGN (with gui-B2): q documented "preserved unchanged" (angle_filtering.py:231); model fits scalar q_list[0]; slicing changes fitted q for APS-old + phi filtering.
- optescape-B1 -> DESIGN: honest fix adds L2 sweeps => individual-mode numerics change.
- cli-B10 -> DESIGN: `from xpcsjax.cli import main` returns the module; fix changes public __all__.
- Covariance/DOF group (optwrap-B5, optstrat-B4/B5/B8, opthet-B9) stays CONFIRMED: popt and chi² unchanged, fixes align reported covariance/DOF with the in-code contract (covariance.py:5-9; wrapper_stratified_route.py:401-438). Flagged to maintainer as uncertainty-changing.
- cli-B3 re-run by lead on locked env (jax/jaxlib 0.11.1): fatal "Unknown flag in XLA_FLAGS" confirmed.
- optstrat-B2 checked against the 2026-06-18 audit (frozen-scaling reindex ruled correct): different mechanism (weighted sigma gather); stands.
- Duplicates: cli-B1 = config-B1; ioviz-B2 = gui-B4. data-B10 fixed coherently with config-B1 (file_path wins).

## ROUND 1 TALLY (126 claims)
CONFIRMED 91 (89 distinct fixes) | DESIGN 23 | REJECTED 12
optrest 4/2/0 · config 7/3/1 · core 7/2/1 · optescape 5/2/1 · gui 12/1/1 · cli 8/2/0 · optwrap 8/1/1 · ioviz 6/2/2 · data 14/1/3 · opthet 9/4/0 · optstrat 11/3/2

- NOTE: reachability rule differs between verifiers (config: public method on exported class counts; data: in-package caller required). Lead applies ONE rule when reporting to Codex.

## ROUND 1 FINAL DISPOSITION (after fixing; supersedes the tally above)
FIXED 83 claims (81 distinct fixes) | DESIGN 31 | REJECTED 12  (total 126)

Reclassified CONFIRMED -> DESIGN by the lead while fixing (red test removed, reason left in the test file):
- data-B1 (+ gui-B2): q documented "preserved unchanged"; slicing changes the fitted q.
- optescape-B1: honest fix adds L2 sweeps -> individual-mode numerics change.
- cli-B10: fix changes the public xpcsjax.cli __all__.
- core-B1: sync_time_axis cannot tell a file shorter than end_frame (trim tail) from frame-0 exclusion (trim head); needs a maintainer choice (reject end_frame beyond the file, or propagate the clamp). REAL OPEN RISK: heterodyne time axis shifted, warning only.
- core-B3: time_grid wiring is a documented maintainer deferral (core/models.py compute_g1_batch docstring, "deferred - see PR #34").
- optstrat-B10/B11: fallback-chain STREAMING tier has never run (AttributeError -> CHUNKED); repairing it activates an unexercised solver on very large data. Maintainer: repair or remove.
- opthet-B10 (+ optstrat-B9): regularization.enable is ignored by design in both models (pinned by test_l3_banner_always_on_even_when_regularization_disabled); template ships enable: false + mode: relative, so honouring it changes default behaviour. A fixer applied it; the lead reverted it.

How fixes differ from the ledger proposals above:
- config-B1/cli-B1: legacy pair is synced to file_path only when a full legacy pair is present (two existing tests pin that a lone data_folder_path is left as written). data-B10: loader derives the pair from a truthy file_path (file_path wins).
- optrest-B5/optwrap-B10: new memory.config_memory_fraction(); precedence env NLSQ_MEMORY_FRACTION > config nlsq.memory_fraction > default (template: env "forces" it). heterodyne_memory already honoured the env var; unchanged.
- optescape-B2: _run_nlsq_refinement now returns the solver's real `success` plus a new `adoptable` flag; the keep-better/adoption gates (cmaes_wrapper, core warm-start) use `adoptable`, so WHICH point is adopted is unchanged. diagnostics["refinement_converged"] added. NOT changed: CMAESResult.success is still `cmaes_converged or nlsq_refined` (left alone: success-gate area).
- optescape-B3: TRF-only kwargs are not passed when the method is hybrid_streaming.
- optwrap-B1/B2/B9, ioviz-B1, optstrat-B1/B2 (unsorted-phi family): invariant "per-angle params/theory are in SORTED-unique phi order; data rows are in caller order" + gathers that are the identity for ascending phi. New helper data_prep.sigma_in_sorted_phi_order. HeterodyneStratifiedData gained phi_raw.
- opthet-B1/B2: `individual` mode with non-ascending phi is solved on phi-sorted data and un-permuted by heterodyne_result_builder.restore_input_angle_order (other modes untouched).
- optwrap-B5: only wrapper.py:1759 fixed. The other listed sites (wrapper.py ~3211 sequential, wrapper_stratified_route.py ~578, core.py ~2906, adapter.py ~1234) were NOT verified and NOT changed.
- optwrap-B8: L threaded into the adapter closure and ModelCacheKey.
- optstrat-B4/B5: DOF counts t1 != t2 rows only (+ OOC subtracts fixed params). optstrat-B8: eigvalsh > 0 check in both streaming Hessian blocks (the laminar one was verified by reading only).
- optstrat-B3: preflight probes every parameter before failing (laminar only; the heterodyne mirror, opthet-B8 DESIGN, is unchanged).
- opthet-B4: _safe_std (double-where) — value bit-identical to jnp.std, gradient 0 instead of NaN at zero variance.
- opthet-B9: averaged path rescales covariance/uncertainties off the L3 penalty rows. The individual-mode twin (_build_joint_result ~:3806) was NOT changed (solver parameter count there unverified).
- data-B8: degradations are stored in cache metadata and replayed on a warm hit (instead of skipping the cache write).
- gui-B9/B8: unique temp names via uuid, not mkstemp (which would create 0600 files).
- cli-B5: a parameter neither active nor fixed is plotted at its registry default (fit value not checked).

Uncertainty-changing fixes (popt and chi² unchanged): optwrap-B5, optstrat-B4/B5/B8, opthet-B9.
