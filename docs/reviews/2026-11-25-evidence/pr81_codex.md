# codex review of PR #81
## Findings
### F1: The in-memory engine treats a fully masked linear fit as unweighted
- Must-fix: yes
- Location: xpcsjax/optimization/nlsq/heterodyne_engine_route.py:457
- Scenario: Fit four distinct angles with `loss="linear"` and `weights=np.zeros_like(c2)`. The uniform-weight check passes and the robust-loss check is skipped. The engine fits every pixel as if it had unit weight, then reports `chi_squared=0` and `reduced_chi_squared=0` on the full row count even though there are zero observations.
- Evidence: The probe below reached the deliberately stubbed unweighted solver. The reporting code at xpcsjax/optimization/nlsq/heterodyne_engine_route.py:642 uses the supplied weights for SSR and computes the observation count from shape alone. The same gate also accepts a mixed mask of 0 and `1e-9` because of the default absolute tolerance in `np.allclose`. Probes ran in the worktree at `35b227c25c12462b70b74596a01769e940650992` as recorded by the checkout inspection.
  ```sh
  rtk uv run --no-sync python -B -c 'import numpy as np
  from unittest.mock import patch
  from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
  from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
  from xpcsjax.optimization.nlsq.heterodyne_adapter import NLSQAdapter
  from xpcsjax.optimization.nlsq.heterodyne_engine_route import fit_two_component_via_engine
  from xpcsjax.optimization.nlsq.heterodyne_result_builder import build_hybrid_streaming_result
  from xpcsjax.optimization.nlsq.heterodyne_stratified_data import build_heterodyne_stratified_data
  from xpcsjax.optimization.nlsq.strategies.heterodyne_hybrid_streaming import build_heterodyne_pointwise_model
  m = HeterodyneModel.from_config(dict(analyzer_parameters=dict(dt=0.1,start_frame=1,end_frame=12,scattering=dict(wavevector_q=0.01))))
  c = np.ones((4,12,12))
  with patch.object(NLSQAdapter,"fit",side_effect=RuntimeError("solver reached")):
      try:
          fit_two_component_via_engine(m,c,np.array([0.,45.,90.,135.]),NLSQConfig(loss="linear"),np.zeros_like(c))
      except Exception as e:
          print("engine with zero weights:", type(e).__name__, str(e))
  p = np.asarray(m.param_manager.get_initial_values())
  r = build_hybrid_streaming_result(model=m,popt=p,pcov=np.eye(len(p)),info=dict(ssr=1.0,n_data_points=len(p)),phi_angles=np.array([0.,90.,135.]))
  print("reduced chi2 on exactly p observations:", r.reduced_chi_squared)
  w = np.ones_like(c)
  w[0] = 0
  sd = build_heterodyne_stratified_data(m,c,np.array([0.,0.,90.,135.]),w)
  meta = build_heterodyne_pointwise_model(stratified_data=sd,model=m,physical_param_names=list(m.param_manager.varying_names))[-1]
  print("duplicate angle observations:",meta["n_contributing_points"],"expected:",3*11*10)'
  ```
- Suggested fix: Reject masks before the uniformity shortcut and take the existing weighted fallback. Restrict the linear shortcut to strictly positive uniform weights.

### F2: Large masked fits report finite reduced chi-squared at zero data DOF
- Must-fix: yes
- Location: xpcsjax/optimization/nlsq/heterodyne_result_builder.py:816
- Scenario: Mask a stratified-LS fit down to exactly as many observations as its reporting parameter count. For SSR 1 and the MSE fallback, the result says reduced chi-squared is 1 even though it is undefined. The stratified host covariance recompute can also return finite covariance when the number of contributing rows equals the fitted parameter count and the remaining Jacobian is full rank.
- Evidence: The F1 command calls the shared result builder with `n_data_points == len(popt)` and prints `1.0`. Both large-data callers now pass contributing rows as `n_data_points` and discard the total count before reporting. The stratified log independently floors the denominator at xpcsjax/optimization/nlsq/heterodyne_stratified_ls.py:1361, and host covariance does the same at xpcsjax/optimization/nlsq/covariance.py:148.
- Suggested fix: Thread both counts through the shared builder and emit NaN for nonpositive masked reporting DOF. Guard the stratified covariance recompute at nonpositive masked fitted DOF and use the same reported statistic in the log. Preserve the legacy all-positive behavior.

### F3: Duplicate phi rows pick up another measurement's mask on both large-data paths
- Must-fix: yes
- Location: xpcsjax/optimization/nlsq/strategies/heterodyne_hybrid_streaming.py:360
- Scenario: Use `phi=[0,0,90,135]`, 12 frames, and zero the weights of only the first measurement. Both measurements at 0 degrees become masked, leaving 220 counted rows instead of 330. The same misassigned sigma goes into the weighted solve.
- Evidence: The F1 command passed those masks to the shared pointwise builder and printed `220 expected: 330`. The sigma slabs are stored by original measurement index. With duplicates, xpcsjax/optimization/nlsq/data_prep.py:103 leaves that axis in input order, but the pointwise builder gathers it with sorted-unique phi slot indices. The duplicate-angle test at tests/review_r1/test_maintainer_decisions.py:302 uses `weights=None`.
- Suggested fix: Align sigma with the original measurement indices before applying the pointwise row mask or shuffle. Continue sharing the unique phi slot for scaling.

### F4: Hybrid-streaming reduced chi-squared includes the error of masked pixels
- Must-fix: yes
- Location: xpcsjax/optimization/nlsq/strategies/heterodyne_hybrid_streaming.py:1040
- Scenario: After fitting, take a prediction that agrees with all positive-weight pixels and differs by 10 at one zero-weight pixel. That pixel adds 100 to the reported reduced chi-squared numerator even though the new denominator excludes it. The in-memory and stratified-LS objectives give that pixel zero contribution.
- Evidence: The streaming optimizer receives sigma at xpcsjax/optimization/nlsq/strategies/heterodyne_hybrid_streaming.py:999, the new observation count is threaded at line 1034, and line 1040 sums squared errors across the entire support without sigma. The shared builder consumes that same SSR at xpcsjax/optimization/nlsq/heterodyne_result_builder.py:782 and :829.
- Suggested fix: For masked hybrid fits, pass a separate SSR for reduced chi-squared that applies the fit's sigma mask and weighting. Keep the legacy raw `chi_squared` field and the all-positive results as specified.

### F5: Legacy regularization settings can override the Layer 3 off switch
- Must-fix: yes
- Location: xpcsjax/optimization/nlsq/heterodyne_config.py:1027
- Scenario: Set flat `regularization_mode="adaptive"` together with nested `anti_degeneracy.regularization.enable=false`. Heterodyne retains the flat mode and the individual joint path still adds L3 rows. On laminar hybrid streaming, `enable_group_variance_regularization=true` in the streaming config likewise survives the nested false flag and is sent to the optimizer.
- Evidence: The command below prints `L3 adaptive`. The heterodyne individual joint builder gates L3 on the resulting mode at xpcsjax/optimization/nlsq/heterodyne_core.py:3396. In the laminar sibling, xpcsjax/optimization/nlsq/strategies/hybrid_streaming.py:567 reads the flat flag, xpcsjax/optimization/nlsq/strategies/hybrid_streaming_layers.py:343 skips construction but returns the input flag at line 566, and xpcsjax/optimization/nlsq/strategies/hybrid_streaming.py:828 passes it to the optimizer.
  ```sh
  rtk uv run --no-sync python -B -c 'import numpy as np
  from xpcsjax.data.xpcs_loader import XPCSDataLoader
  from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
  loader = XPCSDataLoader.__new__(XPCSDataLoader)
  loader.analyzer_config = dict(start_frame=1,end_frame=np.int64(100),dt=0.1)
  trimmed = loader._apply_frame_slicing_to_selected_q(np.ones((1,80,80)))
  print("clamp", trimmed.shape, loader._applied_end_frame, loader._frames_clamped_at_end())
  ad = dict(regularization=dict(enable=False, mode="relative"))
  print("L3", NLSQConfig.from_dict(dict(regularization_mode="adaptive", anti_degeneracy=ad)).regularization_mode)'
  ```
- Suggested fix: Give an explicit nested false precedence in both paths and preserve the legacy settings when the nested switch is omitted. Add mixed-config cases to the current flag tests.

### F6: Programmatic NumPy frame counts lose the tail-clamp metadata
- Must-fix: yes
- Location: xpcsjax/data/xpcs_loader.py:1047
- Scenario: Load an 80-frame HDF5 file with caching disabled and a config dictionary containing `start_frame=1`, `end_frame=np.int64(100)`, and `dt=0.1`. The loader clamps to 80 frames but omits the metadata. The heterodyne model then keeps the last 80 of the configured 100 time points, shifting every fitted time by 2 seconds.
- Evidence: The F5 command prints `clamp (1, 80, 80) 80 0` after logging that it clamped from 100 to 80. The dictionary loader keeps the supplied analyzer values at xpcsjax/data/xpcs_loader.py:477 and :507, and the heterodyne dispatcher defaults absent clamp metadata to zero at xpcsjax/optimization/nlsq/__init__.py:800.
- Suggested fix: Accept integer scalars implementing the index protocol and return a Python int for the metadata. Add a cache-disabled loader regression with a NumPy integer request.

### F7: The unsupported-streaming regression test only checks the presence of a string
- Must-fix: no
- Location: tests/review_r1/test_maintainer_decisions.py:450
- Scenario: Make the new unsupported-config guard unreachable while leaving its body intact. The old generic streaming attempt returns, and this test still passes.
- Evidence: The only assertion searches `inspect.getsource(fallback_chain)` for a literal. The separately updated soft-failure test supplies `hybrid_normalize=True` and exercises the supported-config branch.
- Suggested fix: Pass a config without the required attributes, assert the streaming stub is never called, and assert the following tier runs.

## Closed-as-intended items you dispute
None.
## Verdict
Must-fix count: 6
END_OF_REPORT
